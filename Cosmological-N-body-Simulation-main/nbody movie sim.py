#!/usr/bin/env python3
"""
nbody_movie_sim.py
-------------------
Modified PM simulation that saves 1000 lightweight snapshots
for 3D movie rendering.  Snapshots are saved as compressed .npz
(positions only) to avoid disk overhead of full binary headers.

Run this INSTEAD of the original simulation script, or call
save_movie_snapshot() from inside your existing loop.

Snapshot format:  snapshots/frame_{:04d}.npz
  rra   : float32 (MM, 3)  — positions in grid units
  aa    : float64 scalar   — scale factor
  z     : float64 scalar   — redshift
"""

import os, sys, struct, inspect
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
# PARAMETERS  (keep identical to your original run)
# =====================================================================
par = PyParams(
    N1=128, N2=128, N3=128,
    Nbin=10,
    LL=0.14,
    NF=1,
    vaa=1/200.0,
    delta_aa=0.004
)

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
SEED    = -100012

N_MOVIE_FRAMES = 1000          # total frames to produce
SNAP_DIR       = "snapshots"   # output directory
os.makedirs(SNAP_DIR, exist_ok=True)

# =====================================================================
# COSMOLOGY
# =====================================================================
def Hf(a):
    return np.sqrt(vomegam * a**-3
                   + (1.0 - vomegam - vomegalam) * a**-2
                   + vomegalam)

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
# SNAPSHOT SAVER  (lightweight — positions + scale factor only)
# =====================================================================
def save_movie_snapshot(frame_idx, rra_flat, aa, MM):
    z    = 1.0 / aa - 1.0
    path = os.path.join(SNAP_DIR, f"frame_{frame_idx:04d}.npz")
    np.savez_compressed(
        path,
        rra=rra_flat.reshape(MM, 3).astype(np.float32),
        aa=np.float64(aa),
        z=np.float64(z)
    )

# =====================================================================
# INITIAL CONDITIONS
# =====================================================================
def load_power_spectrum():
    df = pd.read_csv(
        os.path.join(current_dir, "data", "pmf_1nG-2.csv"),
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
# BUILD SCHEDULE OF 1000 FRAME TIMES
# Scale factor goes from a_init (z=199) to a=1 (z=0)
# We space frames uniformly in scale factor so that z=0 clustering
# is well-sampled.  You can switch to log-spacing for early universe.
# =====================================================================
a_final   = 1.0
# Uniform spacing in scale factor
a_frames  = np.linspace(a_init, a_final, N_MOVIE_FRAMES)

# --- alternative: log spacing gives more frames at high-z ---
# a_frames = np.logspace(np.log10(a_init), np.log10(a_final), N_MOVIE_FRAMES)

print(f"Frame schedule: {N_MOVIE_FRAMES} frames")
print(f"  a: {a_frames[0]:.5f} → {a_frames[-1]:.5f}")
print(f"  z: {1/a_frames[0]-1:.1f} → {1/a_frames[-1]-1:.1f}")

# =====================================================================
# MAIN INTEGRATION LOOP  (KDK leapfrog)
# =====================================================================
aa         = a_init
frame_idx  = 0
next_frame = a_frames[frame_idx]

# Save frame 0  (IC)
save_movie_snapshot(frame_idx, rra_flat, aa, MM)
frame_idx += 1
print(f"  Frame {frame_idx-1:04d}  z={1/aa-1:.2f}  a={aa:.5f}")

total_steps = 0

while frame_idx < N_MOVIE_FRAMES:

    next_a_target = a_frames[frame_idx]

    # integrate up to the next frame target
    while aa < next_a_target:
        delta_aa_step = float(par.delta_aa) * (a_init / aa)**0.5
        delta_aa_step = min(delta_aa_step, next_a_target - aa)

        a_mid = aa + 0.5 * delta_aa_step
        a_end = aa + delta_aa_step

        # --- half-kick ---
        _, phi_old   = compute_phi(rra_flat, MM, Nx, Ny, Nz,
                                   rho_b_inv, LL_box, vol)
        py_update_v(MM, Nx, Ny, Nz, aa, 0.5 * delta_aa_step,
                    vomegam, Hf(aa), LL_box,
                    rra_flat, vva_flat, phi_old.ravel())

        # --- drift ---
        dt_drift = delta_aa_step / (a_mid**2 * Hf(a_mid))
        py_update_x(MM, Nx, Ny, Nz, a_mid, dt_drift, dt_drift,
                    rra_flat, vva_flat)
        periodic_wrap(rra_flat, Nx, Ny, Nz, MM)

        # --- half-kick ---
        _, phi_new   = compute_phi(rra_flat, MM, Nx, Ny, Nz,
                                   rho_b_inv, LL_box, vol)
        py_update_v(MM, Nx, Ny, Nz, a_end, 0.5 * delta_aa_step,
                    vomegam, Hf(a_end), LL_box,
                    rra_flat, vva_flat, phi_new.ravel())

        aa = a_end
        total_steps += 1

    # --- save snapshot ---
    save_movie_snapshot(frame_idx, rra_flat, aa, MM)

    if frame_idx % 50 == 0 or frame_idx == N_MOVIE_FRAMES - 1:
        print(f"  Frame {frame_idx:04d}/{N_MOVIE_FRAMES}  "
              f"z={1/aa-1:.3f}  a={aa:.5f}  "
              f"total_steps={total_steps}")

    frame_idx += 1

print(f"\nDone. {frame_idx} snapshots saved to '{SNAP_DIR}/'")
print(f"Total integrator steps: {total_steps}")