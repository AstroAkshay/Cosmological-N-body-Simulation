#!/usr/bin/env python3
"""
nbody_movie_sim_full.py
-----------------------
PM N-body simulation that saves 1000 outputs per run:

  snapshots/frame_{:04d}.npz          — lightweight (positions only)
  output/output.nbody_{Z:.3f}         — full binary (matches fof.c reader)

The .npz files feed render_movie.py (fast).
The output.nbody_ files feed the FOF halo finder (fof.c).

Binary layout of output.nbody_ (matches write_output in original code):
  [dummy=0][io_header 256B][dummy=0]
  [dummy=0][rra  MM*3 float32 AoS][dummy=0]
  [dummy=0][vva  MM*3 float32 AoS][dummy=0]

output_flag=1  → positions in N-body grid units, velocities in code units
output_flag=0  → positions in kpc/h, velocities in km/s  (Gadget-style)

Set OUTPUT_FLAG below.  FOF code (fof.c) expects output_flag=1.
"""

import os, sys, struct
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

os.environ["OMP_NUM_THREADS"] = "22"
current_dir = os.getcwd()
sys.path.insert(0, current_dir)

from nbodysim.hermitian_wrapper import generate_hermitian_grid
from nbodysim.params             import PyParams
from nbodysim.fftw_helper        import fft3d_r2c_py, fft3d_c2r_py
from nbodysim.calpow_module      import calpow
from nbodysim.gradphi            import grad_phi_py
from nbodysim.zel_move_wrap      import py_Zel_move_gradphi
from nbodysim.cic_wrap           import cic_py
from nbodysim.update_position    import py_update_x
from nbodysim.update_velocity    import py_update_v
from nbodysim.get_phi_k          import apply_inverse_laplacian

# =====================================================================
# USER PARAMETERS
# =====================================================================
par = PyParams(
    N1=128, N2=128, N3=128,
    Nbin=10,
    LL=0.14,
    NF=1,
    vaa=1/200.0,
    delta_aa=0.004
)

# Output flags
N_MOVIE_FRAMES = 1000          # number of output times (frames + nbody files)
OUTPUT_FLAG    = 1             # 1 = grid units (needed by fof.c)
                               # 0 = kpc/h + km/s  (Gadget-style)
SNAP_DIR       = "snapshots"   # .npz output directory
NBODY_DIR      = "output"      # output.nbody_ directory
SEED           = -100012

# =====================================================================
# DERIVED QUANTITIES
# =====================================================================
Nx, Ny, Nz = int(par.N1), int(par.N2), int(par.N3)
LL_box     = float(par.LL)
vomegam    = float(par.vomegam)
vomegalam  = float(par.vomegalam)
vhh        = float(par.vhh)
vomegab    = float(par.vomegab)
sigma_8    = float(par.sigma_8_present)

vol     = Nx * Ny * Nz * LL_box**3
Npoints = (Nx // par.NF) * (Ny // par.NF) * (Nz // par.NF)
MM      = Npoints

os.makedirs(SNAP_DIR,  exist_ok=True)
os.makedirs(NBODY_DIR, exist_ok=True)

# =====================================================================
# COSMOLOGY
# =====================================================================
def Hf(a):
    return np.sqrt(vomegam * a**-3
                   + (1.0 - vomegam - vomegalam) * a**-2
                   + vomegalam)

# =====================================================================
# PERIODIC WRAP  (float32-safe)
# =====================================================================
def periodic_wrap(rra_flat, Nx, Ny, Nz, MM):
    rra_view = rra_flat.reshape(MM, 3)
    for dim, N in enumerate([Nx, Ny, Nz]):
        x = rra_view[:, dim]
        bad = ~np.isfinite(x)
        if bad.any():
            x[bad] = 0.0
        x -= N * np.floor(x / N)
        x[:] = np.clip(x, 0.0, N - 0.001)
        rra_view[:, dim] = x

# =====================================================================
# GRAVITATIONAL POTENTIAL
# =====================================================================
def compute_phi(rra_flat, MM, Nx, Ny, Nz, rho_b_inv, LL_box, vol):
    ro      = cic_py(rra_flat.reshape(MM, 3), Nx, Ny, Nz, rho_b_inv)
    delta_g = ro / ro.mean() - 1.0
    delta_k = fft3d_r2c_py(delta_g * LL_box**3)
    phi_k   = delta_k.copy()
    apply_inverse_laplacian(phi_k, Nx * LL_box, vol)
    phi  = fft3d_c2r_py(phi_k, Nx, Ny, Nz)
    phi /= (Nx * Ny * Nz)
    phi /= LL_box**2
    return ro, phi

# =====================================================================
# SNAPSHOT SAVER  —  lightweight .npz  (for renderer)
# =====================================================================
def save_movie_snapshot(frame_idx, rra_flat, aa, MM):
    z    = 1.0 / aa - 1.0
    path = os.path.join(SNAP_DIR, f"frame_{frame_idx:04d}.npz")
    np.savez_compressed(
        path,
        rra = rra_flat.reshape(MM, 3).astype(np.float32),
        aa  = np.float64(aa),
        z   = np.float64(z),
    )

# =====================================================================
# FULL BINARY OUTPUT  —  output.nbody_  (for fof.c)
# Matches write_output() in the original simulation exactly.
#
# Header layout (256 bytes, little-endian):
#   offset  0 : 6 × int32   npart[6]
#   offset 24 : 6 × float64 mass[6]
#   offset 72 : float64     time  (= scale factor a)
#   offset 80 : float64     redshift
#   offset 88 : int32       flag_sfr
#   offset 92 : int32       flag_feedback
#   offset 96 : 6 × int32   npartTotal[6]
#   offset 120: int32       flag_cooling
#   offset 124: int32       num_files
#   offset 128: float64     BoxSize  (kpc/h)
#   offset 136: float64     Omega0
#   offset 144: float64     OmegaLambda
#   offset 152: float64     HubbleParam
#   offset 160: float64     Omegab
#   offset 168: float64     sigma_8_present
#   offset 176: int32       Nx
#   offset 180: int32       Ny
#   offset 184: int32       Nz
#   offset 188: float32     LL
#   offset 192: int32       output_flag
#   offset 196: int32       in_flag
#   offset 200: int32       seed
#   offset 204: 52 bytes    padding  (fills to 256)
# =====================================================================
def write_nbody_output(frame_idx, rra_flat, vva_flat, aa, output_flag):
    redshift = 1.0 / aa - 1.0
    z_str    = f"{redshift:.3f}"
    fname    = os.path.join(NBODY_DIR, f"output.nbody_{z_str}")

    rho_crit = 2.775e11   # M_sun h² / Mpc³
    DM_m     = vomegam * rho_crit * (Nx * LL_box)**3 / MM / 1e10

    # ---- build 256-byte header ----
    header_body = struct.pack(
        '<'
        '6i'   # npart[6]
        '6d'   # mass[6]
        'd'    # time
        'd'    # redshift
        'i'    # flag_sfr
        'i'    # flag_feedback
        '6i'   # npartTotal[6]
        'i'    # flag_cooling
        'i'    # num_files
        'd'    # BoxSize  (kpc/h)
        'd'    # Omega0
        'd'    # OmegaLambda
        'd'    # HubbleParam
        'd'    # Omegab
        'd'    # sigma_8_present
        'i'    # Nx
        'i'    # Ny
        'i'    # Nz
        'f'    # LL
        'i'    # output_flag
        'i'    # in_flag
        'i',   # seed
        # npart
        0, MM, 0, 0, 0, 0,
        # mass
        0.0, DM_m, 0.0, 0.0, 0.0, 0.0,
        # time, redshift
        float(aa), float(redshift),
        # flag_sfr, flag_feedback
        0, 0,
        # npartTotal
        0, MM, 0, 0, 0, 0,
        # flag_cooling, num_files
        0, 1,
        # BoxSize
        float(Nx * LL_box * 1000.0 * vhh),
        float(vomegam),
        float(vomegalam),
        float(vhh),
        float(vomegab),
        float(sigma_8),
        Nx, Ny, Nz,
        float(LL_box),
        int(output_flag), 1,
        int(SEED),
    )

    assert len(header_body) == 204, f"Header body {len(header_body)} != 204"
    header = header_body + b'\x00' * 52   # pad to 256 bytes
    assert len(header) == 256

    # ---- prepare position / velocity arrays ----
    rra = rra_flat.reshape(MM, 3).astype(np.float64).copy()
    vva = vva_flat.reshape(MM, 3).astype(np.float64).copy()

    # float32-safe periodic wrap on positions
    for dim, N in enumerate([Nx, Ny, Nz]):
        rra[:, dim] -= N * np.floor(rra[:, dim] / N)
        rra[:, dim]  = np.clip(rra[:, dim], 0.0, N - 0.001)

    if output_flag == 0:
        # Convert to physical units (Gadget-style)
        rra = (rra * LL_box * 1000.0 * vhh).astype(np.float32)
        vva = (vva * LL_box * vhh * 100.0 / aa).astype(np.float32)
    else:
        # Stay in grid units (what fof.c reads with output_flag=1)
        rra = rra.astype(np.float32)
        vva = vva.astype(np.float32)

    # Final float32-level clamp (catches rounding at the boundary)
    if output_flag == 1:
        rra[:, 0] = np.clip(rra[:, 0], 0.0, np.float32(Nx - 0.001))
        rra[:, 1] = np.clip(rra[:, 1], 0.0, np.float32(Ny - 0.001))
        rra[:, 2] = np.clip(rra[:, 2], 0.0, np.float32(Nz - 0.001))

    # ---- write ----
    dummy = struct.pack('<i', 0)
    with open(fname, 'wb') as f:
        # Header block
        f.write(dummy); f.write(header); f.write(dummy)
        # Position block  (AoS: x0 y0 z0  x1 y1 z1 …)
        f.write(dummy)
        f.write(rra.tobytes())
        f.write(dummy)
        # Velocity block
        f.write(dummy)
        f.write(vva.tobytes())
        f.write(dummy)

    return fname, redshift, DM_m


# =====================================================================
# OPTIONAL: header verification  (called once per output for safety)
# =====================================================================
def verify_header(fname):
    with open(fname, 'rb') as f:
        f.read(4)
        raw = f.read(256)

    npart    = struct.unpack_from('<6i', raw,   0)
    mass     = struct.unpack_from('<6d', raw,  24)
    time_a   = struct.unpack_from('<d',  raw,  72)[0]
    redshift = struct.unpack_from('<d',  raw,  80)[0]
    BoxSize  = struct.unpack_from('<d',  raw, 128)[0]
    Nx_h     = struct.unpack_from('<i',  raw, 176)[0]
    oflag    = struct.unpack_from('<i',  raw, 192)[0]
    seed_h   = struct.unpack_from('<i',  raw, 200)[0]

    print(f"    [verify] npart[1]={npart[1]}  "
          f"mass[1]={mass[1]:.3e}  "
          f"a={time_a:.5f}  z={redshift:.3f}  "
          f"Box={BoxSize:.1f}kpc/h  Nx={Nx_h}  "
          f"flag={oflag}  seed={seed_h}")


# =====================================================================
# INITIAL CONDITIONS
# =====================================================================
def load_power_spectrum():
    df = pd.read_csv(
        os.path.join(current_dir, "data", "P_CDM(199).csv"),
        header=None, names=["k", "Pk"]
    )
    return df["k"].values, df["Pk"].values

k_linear, pk_linear = load_power_spectrum()
Pk_spline = CubicSpline(k_linear, pk_linear, bc_type="natural")

delta_k_raw = generate_hermitian_grid(Nx, Ny, Nz, Nx * LL_box, Pk_spline)
delta_k_raw *= np.sqrt(vol)
delta_k_raw[0, 0, 0] = 0.0

delta_k = delta_k_raw[:, :, :Nz // 2 + 1].astype(np.complex128)

disp_x = np.zeros_like(delta_k, dtype=np.complex128)
disp_y = np.zeros_like(delta_k, dtype=np.complex128)
disp_z = np.zeros_like(delta_k, dtype=np.complex128)

Cx, Cy, Cz = 2 * np.pi / Nx, 2 * np.pi / Ny, 2 * np.pi / Nz
for i, disp in enumerate([disp_x, disp_y, disp_z]):
    grad_phi_py(i, delta_k, disp, Nx, Ny, Nz, Cx, Cy, Cz, vol)

disp_x_ravel = fft3d_c2r_py(disp_x, Nx, Ny, Nz).ravel()
disp_y_ravel = fft3d_c2r_py(disp_y, Nx, Ny, Nz).ravel()
disp_z_ravel = fft3d_c2r_py(disp_z, Nx, Ny, Nz).ravel()

rra = np.zeros((Npoints, 3), dtype=np.float64)
vva = np.zeros_like(rra)

a_init = float(par.vaa)
vfac   = a_init**2 * Hf(a_init)

py_Zel_move_gradphi(
    vfac, rra.ravel(), vva.ravel(),
    disp_x_ravel, disp_y_ravel, disp_z_ravel,
    Nx, Ny, Nz, par.NF, LL_box
)

rra_flat = rra.ravel()
vva_flat = vva.ravel()
periodic_wrap(rra_flat, Nx, Ny, Nz, Npoints)

rho_b_inv = (Nx * Ny * Nz) / Npoints

# =====================================================================
# FRAME SCHEDULE
# 1000 output times uniformly spaced in scale factor  a ∈ [a_init, 1]
# Swap to log-spacing for denser sampling at high redshift:
#   a_frames = np.logspace(np.log10(a_init), 0.0, N_MOVIE_FRAMES)
# =====================================================================
a_final  = 1.0
a_frames = np.linspace(a_init, a_final, N_MOVIE_FRAMES)

print(f"Frame schedule: {N_MOVIE_FRAMES} frames")
print(f"  a : {a_frames[0]:.5f} → {a_frames[-1]:.5f}")
print(f"  z : {1/a_frames[0]-1:.1f} → {1/a_frames[-1]-1:.1f}")
print(f"  Output dirs:  '{SNAP_DIR}/'  and  '{NBODY_DIR}/'")

# =====================================================================
# SAVE FRAME 0  (initial conditions)
# =====================================================================
aa        = a_init
frame_idx = 0

save_movie_snapshot(frame_idx, rra_flat, aa, MM)

fname, z0, DM_m0 = write_nbody_output(frame_idx, rra_flat, vva_flat,
                                       aa, OUTPUT_FLAG)
print(f"  Frame {frame_idx:04d}  z={z0:.3f}  a={aa:.5f}  "
      f"→  {os.path.basename(fname)}")
verify_header(fname)

frame_idx += 1
total_steps = 0

# =====================================================================
# MAIN KDK LEAPFROG LOOP
# =====================================================================
while frame_idx < N_MOVIE_FRAMES:

    next_a_target = a_frames[frame_idx]

    # ---- integrate forward to next frame time ----
    while aa < next_a_target:
        delta_aa_step = float(par.delta_aa) * (a_init / aa) ** 0.5
        delta_aa_step = min(delta_aa_step, next_a_target - aa)

        a_mid = aa + 0.5 * delta_aa_step
        a_end = aa + delta_aa_step

        # half-kick
        _, phi_old = compute_phi(rra_flat, MM, Nx, Ny, Nz,
                                 rho_b_inv, LL_box, vol)
        py_update_v(MM, Nx, Ny, Nz, aa, 0.5 * delta_aa_step,
                    vomegam, Hf(aa), LL_box,
                    rra_flat, vva_flat, phi_old.ravel())

        # drift
        dt_drift = delta_aa_step / (a_mid ** 2 * Hf(a_mid))
        py_update_x(MM, Nx, Ny, Nz, a_mid, dt_drift, dt_drift,
                    rra_flat, vva_flat)
        periodic_wrap(rra_flat, Nx, Ny, Nz, MM)

        # half-kick
        _, phi_new = compute_phi(rra_flat, MM, Nx, Ny, Nz,
                                 rho_b_inv, LL_box, vol)
        py_update_v(MM, Nx, Ny, Nz, a_end, 0.5 * delta_aa_step,
                    vomegam, Hf(a_end), LL_box,
                    rra_flat, vva_flat, phi_new.ravel())

        aa = a_end
        total_steps += 1

    # ---- save both outputs at this frame time ----

    # 1) lightweight .npz  (fast render)
    save_movie_snapshot(frame_idx, rra_flat, aa, MM)

    # 2) full binary  (FOF halo finder)
    fname, z_out, _ = write_nbody_output(frame_idx, rra_flat, vva_flat,
                                          aa, OUTPUT_FLAG)

    # ---- progress report ----
    if frame_idx % 50 == 0 or frame_idx == N_MOVIE_FRAMES - 1:
        print(f"  Frame {frame_idx:04d}/{N_MOVIE_FRAMES}  "
              f"z={z_out:.4f}  a={aa:.5f}  "
              f"steps={total_steps}  "
              f"→  {os.path.basename(fname)}")

    frame_idx += 1

# =====================================================================
# DONE
# =====================================================================
print(f"\nDone.")
print(f"  {frame_idx} × .npz snapshots  →  '{SNAP_DIR}/'")
print(f"  {frame_idx} × output.nbody_   →  '{NBODY_DIR}/'")
print(f"  Total integrator steps: {total_steps}")
print()

# Quick size estimate
nbody_size_bytes = MM * 6 * 4 + 256 + 6 * 8   # positions + velocities + header + dummies
total_gb = frame_idx * nbody_size_bytes / 1e9
print(f"  Estimated output.nbody_ total size: ~{total_gb:.1f} GB")
npz_size_bytes = MM * 3 * 4 * 0.4             # ~40% compression typical
total_npz_gb = frame_idx * npz_size_bytes / 1e9
print(f"  Estimated .npz total size:          ~{total_npz_gb:.2f} GB")