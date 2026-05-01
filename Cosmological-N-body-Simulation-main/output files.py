#!/usr/bin/env python3
"""
COSMOLOGICAL PM SIMULATION - WITH FILE OUTPUT
Parameters: N=64^3, NF=1, LL=0.14 Mpc, a_init=1/200 (z=199)
Box size = 64 * 0.14 = 8.96 Mpc
Units: Mpc internally
Windows MINGW64: sizeof(long)=4, all long fields packed as 'i'
"""

# =====================================================
# IMPORTS
# =====================================================
import os
import sys
import struct
import inspect

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import CubicSpline, interp1d

# =====================================================
# CPU THREADS
# =====================================================
os.environ["OMP_NUM_THREADS"] = "22"

# =====================================================
# LOCAL MODULE PATH
# =====================================================
current_dir = os.getcwd()
sys.path.insert(0, current_dir)

# =====================================================
# IMPORT C/CYTHON MODULES
# =====================================================
from nbodysim.hermitian_wrapper import generate_hermitian_grid
from nbodysim.params import PyParams
from nbodysim.fftw_helper import fft3d_r2c_py, fft3d_c2r_py
from nbodysim.calpow_module import calpow
from nbodysim.gradphi import grad_phi_py
from nbodysim.zel_move_wrap import py_Zel_move_gradphi
from nbodysim.cic_wrap import cic_py
from nbodysim.update_position import py_update_x
from nbodysim.update_velocity import py_update_v
from nbodysim.get_phi_k import apply_inverse_laplacian

# =====================================================
# SIMULATION PARAMETERS
#   N1=N2=N3=64, NF=1, LL=0.14 Mpc
#   Box = 64 * 0.14 = 8.96 Mpc
#   a_init = 1/200 = 0.005 (z=199)
# =====================================================
par = PyParams(
    N1=128, N2=128, N3=128,
    Nbin=10,
    LL=0.14,
    NF=1,
    vaa=1/200.0,
    delta_aa=0.004
)

par.print()

Nx, Ny, Nz  = int(par.N1), int(par.N2), int(par.N3)
LL_box      = float(par.LL)
Nbin        = par.Nbin
vomegam     = float(par.vomegam)
vomegalam   = float(par.vomegalam)
vhh         = float(par.vhh)
vomegab     = float(par.vomegab)
sigma_8     = float(par.sigma_8_present)

# Derived
vol      = Nx * Ny * Nz * LL_box**3
Npoints  = (Nx // par.NF) * (Ny // par.NF) * (Nz // par.NF)
MM       = Npoints

SEED = -100012

# =====================================================
# FILENAME HELPER
# =====================================================
def nbody_filename(z):
    return f"output/output.nbody_{z:.3f}"

#==========================================
test = np.random.randn(Nx, Ny, Nz)
roundtrip = fft3d_c2r_py(fft3d_r2c_py(test), Nx, Ny, Nz)
print(f"IFFT normalization factor: {(roundtrip/test).mean():.6f}") 
# If 1.0 → normalized. If 262144.0 → unnormalized (FFTW default)

print(f"vol = {vol:.6f}")   # should be (Nx*LL_box)^3 = 8.96^3 = 719.56 Mpc^3
print(f"Nx*Ny*Nz = {Nx*Ny*Nz}")  # 262144
#========================================

# =====================================================
# PERIODIC WRAP
# Uses floor-based wrap + clamp with float32-safe epsilon
# float32 cannot represent N - 1e-6, rounds back to N
# Use 0.001 which is safely representable in float32
# =====================================================
def periodic_wrap(rra_flat, Nx, Ny, Nz, MM):
    rra_view = rra_flat.reshape(MM, 3)
    for dim, N in enumerate([Nx, Ny, Nz]):
        x = rra_view[:, dim]
        # Fix non-finite values
        bad = ~np.isfinite(x)
        if bad.any():
            print(f"  WARNING: {bad.sum()} non-finite positions "
                  f"in dim {dim}")
            x[bad] = 0.0
        # Floor-based wrap handles any magnitude
        x -= N * np.floor(x / N)
        # Clamp with float32-safe epsilon (1e-6 is NOT enough)
        x[:] = np.clip(x, 0.0, N - 0.001)
        rra_view[:, dim] = x

# =====================================================
# COSMOLOGICAL GROWTH FUNCTIONS
# =====================================================
def Hf(a):
    return np.sqrt(vomegam * a**-3
                   + (1.0 - vomegam - vomegalam) * a**-2
                   + vomegalam)

def Integrandf(a):
    return (a * Hf(a))**-3

def Integralf(a):
    Nsteps = 1000
    aa_arr = np.linspace(1e-5, a, Nsteps)
    return np.trapz(Integrandf(aa_arr), aa_arr)

def Df(a):
    return Hf(a) * Integralf(a) / (Hf(1.0) * Integralf(1.0))

# =====================================================
# WRITE OUTPUT
# Matches C write_output() exactly.
# sizeof(long)=4 on Windows MINGW64 -> all long as 'i'
# Applies float32-safe wrap before writing.
#
# Binary layout:
#   [dummy][header 256B][dummy]
#   [dummy][rra MM*3 floats AoS][dummy]
#   [dummy][vva MM*3 floats AoS][dummy]
# =====================================================
def write_output(fname, seed, output_flag, rra, vva, vaa):
    MM_loc   = rra.shape[0]
    redshift = 1.0 / vaa - 1.0
    rho_crit = 2.775e11
    DM_m     = vomegam * rho_crit * (Nx * LL_box)**3 / MM_loc / 1e10

    header = struct.pack(
        '<'
        '6i'   # npart[6]          long -> int32 on Windows
        '6d'   # mass[6]
        'd'    # time
        'd'    # redshift
        'i'    # flag_sfr
        'i'    # flag_feedback
        '6i'   # npartTotal[6]     long -> int32 on Windows
        'i'    # flag_cooling
        'i'    # num_files
        'd'    # BoxSize           kpc/h: N1*LL*1000*vhh
        'd'    # Omega0
        'd'    # OmegaLambda
        'd'    # HubbleParam
        'd'    # Omegab
        'd'    # sigma_8_present
        'i'    # Nx                long -> int32 on Windows
        'i'    # Ny
        'i'    # Nz
        'f'    # LL
        'i'    # output_flag
        'i'    # in_flag
        'i',   # seed              long -> int32 on Windows
        0, MM_loc, 0, 0, 0, 0,
        0.0, DM_m, 0.0, 0.0, 0.0, 0.0,
        float(vaa), float(redshift),
        0, 0,
        0, MM_loc, 0, 0, 0, 0,
        0, 1,
        float(Nx * LL_box * 1000.0 * vhh),
        float(vomegam),
        float(vomegalam),
        float(vhh),
        float(vomegab),
        float(sigma_8),
        Nx, Ny, Nz,
        float(LL_box),
        int(output_flag), 1,
        int(seed),
    )

    assert len(header) == 204, f"Header {len(header)} != 204"
    header = header + b'\x00' * 52

    # Apply float32-safe wrap before casting
    # This prevents boundary particles (exactly N) from segfaulting C code
    rra_safe = rra.astype(np.float64).copy()
    for dim, N in enumerate([Nx, Ny, Nz]):
        rra_safe[:, dim] -= N * np.floor(rra_safe[:, dim] / N)
        rra_safe[:, dim]  = np.clip(rra_safe[:, dim], 0.0, N - 0.001)

    rra_out = rra_safe.astype(np.float32)
    vva_out = vva.astype(np.float32).copy()

    if output_flag == 0:
        rra_out = (rra_out * LL_box * 1000.0 * vhh).astype(np.float32)
        vva_out = (vva_out * LL_box * vhh * 100.0 / vaa).astype(np.float32)

    # Final float32-level clamp — catches any rounding after float32 cast
    rra_out[:, 0] = np.clip(rra_out[:, 0], 0.0, np.float32(Nx - 0.001))
    rra_out[:, 1] = np.clip(rra_out[:, 1], 0.0, np.float32(Ny - 0.001))
    rra_out[:, 2] = np.clip(rra_out[:, 2], 0.0, np.float32(Nz - 0.001))

    dummy = struct.pack('<i', 0)
    with open(fname, 'wb') as f:
        f.write(dummy); f.write(header); f.write(dummy)
        f.write(dummy)
        for ii in range(MM_loc):
            f.write(rra_out[ii].tobytes())
        f.write(dummy)
        f.write(dummy)
        for ii in range(MM_loc):
            f.write(vva_out[ii].tobytes())
        f.write(dummy)

    print(f"  [write_output] {fname}  z={redshift:.3f}  a={vaa:.5f}  "
          f"MM={MM_loc}  DM_m={DM_m:.4e}  flag={output_flag}")

# =====================================================
# VERIFY HEADER
# =====================================================
def verify_header(fname):
    with open(fname, 'rb') as f:
        f.read(4)
        raw = f.read(256)

    npart      = struct.unpack_from('<6i', raw,   0)
    mass       = struct.unpack_from('<6d', raw,  24)
    time       = struct.unpack_from('<d',  raw,  72)[0]
    redshift   = struct.unpack_from('<d',  raw,  80)[0]
    npartTotal = struct.unpack_from('<6i', raw,  96)
    BoxSize    = struct.unpack_from('<d',  raw, 128)[0]
    Nx_h       = struct.unpack_from('<i',  raw, 176)[0]
    Ny_h       = struct.unpack_from('<i',  raw, 180)[0]
    Nz_h       = struct.unpack_from('<i',  raw, 184)[0]
    LL_h       = struct.unpack_from('<f',  raw, 188)[0]
    oflag      = struct.unpack_from('<i',  raw, 192)[0]
    in_flag    = struct.unpack_from('<i',  raw, 196)[0]
    seed_h     = struct.unpack_from('<i',  raw, 200)[0]

    print(f"  --- verify: {fname} ---")
    print(f"  npart[1]      = {npart[1]}")
    print(f"  npartTotal[1] = {npartTotal[1]}")
    print(f"  mass[1]       = {mass[1]:.4e}  (10^10 M_sun/h)")
    print(f"  time(a)       = {time:.5f}")
    print(f"  redshift      = {redshift:.3f}")
    print(f"  BoxSize       = {BoxSize:.4f} kpc/h")
    print(f"  Nx,Ny,Nz      = {Nx_h},{Ny_h},{Nz_h}")
    print(f"  LL            = {LL_h:.6f} Mpc")
    print(f"  output_flag   = {oflag}")
    print(f"  in_flag       = {in_flag}")
    print(f"  seed          = {seed_h}")

# =====================================================
# POSITION RANGE DIAGNOSTIC
# =====================================================
def check_positions(fname, MM, Nx, label=""):
    with open(fname, 'rb') as f:
        f.read(4); f.read(256); f.read(4)
        f.read(4)
        pos = np.frombuffer(
            f.read(MM * 3 * 4), dtype=np.float32
        ).reshape(MM, 3)

    bad = (pos < 0).any() or (pos >= Nx).any()
    print(f"  {label}  "
          f"x=[{pos[:,0].min():.4f},{pos[:,0].max():.4f}]  "
          f"y=[{pos[:,1].min():.4f},{pos[:,1].max():.4f}]  "
          f"z=[{pos[:,2].min():.4f},{pos[:,2].max():.4f}]  "
          f"n_bad={((pos<0)|(pos>=Nx)).sum()}  "
          f"{'*** BAD ***' if bad else 'OK'}")

# =====================================================
# LOAD LINEAR POWER SPECTRUM (at z=199)
# =====================================================
def load_power_spectrum():
    module_dir = os.path.dirname(inspect.getfile(load_power_spectrum))
    df = pd.read_csv(
        os.path.join(module_dir, "data", "P_CDM(199).csv"),
        header=None, names=["k", "Pk"]
    )
    return df["k"].values, df["Pk"].values

k_linear, pk_linear = load_power_spectrum()
Pk_spline = CubicSpline(k_linear, pk_linear, bc_type="natural")

# =====================================================
# INITIAL CONDITIONS (ZEL'DOVICH APPROXIMATION)
# =====================================================
delta_k_raw = generate_hermitian_grid(Nx, Ny, Nz, Nx * LL_box, Pk_spline)
delta_k_raw *= np.sqrt(vol)
delta_k_raw[0, 0, 0] = 0.0

delta_k = delta_k_raw[:, :, :Nz // 2 + 1].astype(np.complex128)
delta_x = fft3d_c2r_py(delta_k / vol, Nx, Ny, Nz)

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
H_init = Hf(a_init)
vfac   = a_init**2 * H_init

py_Zel_move_gradphi(
    vfac, rra.ravel(), vva.ravel(),
    disp_x_ravel, disp_y_ravel, disp_z_ravel,
    Nx, Ny, Nz, par.NF, LL_box
)

# IC diagnostics
print(f"IC positions range: "
      f"x=[{rra[:,0].min():.4f},{rra[:,0].max():.4f}]  "
      f"y=[{rra[:,1].min():.4f},{rra[:,1].max():.4f}]  "
      f"z=[{rra[:,2].min():.4f},{rra[:,2].max():.4f}]")
print(f"Any non-finite positions: {(~np.isfinite(rra)).any()}")
print(f"Any non-finite velocities: {(~np.isfinite(vva)).any()}")

# Periodic wrap after IC
rra_flat = rra.ravel()
vva_flat = vva.ravel()
periodic_wrap(rra_flat, Nx, Ny, Nz, Npoints)

# =====================================================
# INITIAL DENSITY & POWER SPECTRUM
# =====================================================
rho_b_inv   = (Nx * Ny * Nz) / Npoints
ro_ini      = cic_py(rra_flat.reshape(Npoints, 3), Nx, Ny, Nz, rho_b_inv)
delta_ini   = ro_ini / ro_ini.mean() - 1.0
delta_k_ini = fft3d_r2c_py(delta_ini * LL_box**3)

# =====================================================
# CIC DECONVOLUTION
# =====================================================
def cic_deconvolution(delta_k, Nx, Ny, Nz):
    fx_norm = np.arange(Nx) / Nx;  fx_norm[Nx//2:] -= 1.0
    fy_norm = np.arange(Ny) / Ny;  fy_norm[Ny//2:] -= 1.0
    fz_norm = np.arange(Nz//2+1) / Nz
    FX, FY, FZ = np.meshgrid(fx_norm, fy_norm, fz_norm, indexing='ij')
    W2 = (np.sinc(FX) * np.sinc(FY) * np.sinc(FZ))**2
    return delta_k / (W2 + 1e-10)

# =====================================================
# HELPER: density -> gravitational potential
# =====================================================
def compute_phi(rra_flat, MM, Nx, Ny, Nz, rho_b_inv, LL_box, vol):
    ro      = cic_py(rra_flat.reshape(MM, 3), Nx, Ny, Nz, rho_b_inv)
    delta_g = ro / ro.mean() - 1.0

    delta_k = fft3d_r2c_py(delta_g * LL_box**3)
    phi_k   = delta_k.copy()

    Lbox_mpc = Nx * LL_box
    apply_inverse_laplacian(phi_k, Lbox_mpc, vol)

    phi = fft3d_c2r_py(phi_k, Nx, Ny, Nz)

    phi /= (Nx * Ny * Nz)   # unnormalized IFFT → Mpc²
    phi /= LL_box**2         # Mpc² → dimensionless (what C gradient expects)

    return ro, phi
# ============================================================
# NORMALIZATION SELF-TEST
# Uses linear theory: phi_rms should match delta_rms / k²_rms
# ============================================================
ro_test  = cic_py(rra_flat.reshape(MM,3), Nx, Ny, Nz, rho_b_inv)
dg_test  = ro_test / ro_test.mean() - 1.0
dk_test  = fft3d_r2c_py(dg_test * LL_box**3)

# Expected phi_k magnitude at k_fundamental:
k1 = 2*np.pi / (Nx * LL_box)
dk_rms = np.abs(dk_test).mean()
phi_expected_rms = dk_rms / k1**2

phi_k_test = dk_test.copy()
apply_inverse_laplacian(phi_k_test, Nx*LL_box, vol)

phi_real_raw    = fft3d_c2r_py(phi_k_test, Nx, Ny, Nz)
phi_div_N3      = phi_real_raw / (Nx*Ny*Nz)
phi_div_N3_vol  = phi_div_N3 / vol
phi_div_N3_LL   = phi_div_N3 / LL_box

print(f"delta_g rms        = {dg_test.std():.4e}")
print(f"delta_k rms        = {dk_rms:.4e} Mpc³")
print(f"phi_k/k² expected  ~ {phi_expected_rms:.4e} Mpc⁵")
print(f"phi /N³            rms = {phi_div_N3.std():.4e}")
print(f"phi /N³/vol        rms = {phi_div_N3_vol.std():.4e}")
print(f"phi /N³/LL_box     rms = {phi_div_N3_LL.std():.4e}")
print(f"Target phi rms         ~ 1e-4 to 1e-3 Mpc²")

# =====================================================
# KDK LEAPFROG TIME INTEGRATION
# =====================================================

print("=== Starting simulation ===")
print(f"    Box = {Nx}x{Ny}x{Nz},  "
      f"L = {Nx*LL_box:.2f} Mpc,  MM = {MM}")

nz = [199, 50,10, 5, 2, 1.5, 1.2, 0.8, 0.4, 0]

Pk_history   = np.zeros((len(nz), Nbin))
a_history    = np.zeros(len(nz))
delta_slices = np.zeros((len(nz), Nx, Ny))

# ---- IC snapshot (z=199) ----
delta_k_ini_deconv          = cic_deconvolution(delta_k_ini, Nx, Ny, Nz)
Pk_ini, kmode_binned_ini, _ = calpow(delta_k_ini_deconv, Nbin,
                                      2*np.pi/LL_box, vol)
Pk_history[0]               = Pk_ini
a_history[0]                = a_init

ro_ini_proj     = np.sum(ro_ini, axis=2) / Nz
delta_slices[0] = ro_ini_proj / ro_ini_proj.mean() - 1.0

write_output(
    nbody_filename(nz[0]),
    seed=SEED, output_flag=1,
    rra=rra_flat.reshape(MM, 3),
    vva=vva_flat.reshape(MM, 3),
    vaa=a_init
)
verify_header(nbody_filename(nz[0]))
check_positions(nbody_filename(nz[0]), MM, Nx, label=f"z={nz[0]}")

aa = a_init

k_min = kmode_binned_ini
idx_min = 0
print(f"Lowest k mode: k={k_min}Mpc⁻¹")

# ---- Time loop ----
for jj in range(1, len(nz)):
    afin = 1.0 / (nz[jj] + 1.0)

    while aa < afin:
        delta_aa_step = par.delta_aa * (a_init / aa)**0.5
        delta_aa_step = min(delta_aa_step, afin - aa)

        a_mid = aa + 0.5 * delta_aa_step
        a_end = aa + delta_aa_step

        # Force at current positions
        _, phi_old   = compute_phi(rra_flat, MM, Nx, Ny, Nz,
                                   rho_b_inv, LL_box, vol)
        phi_old_flat = phi_old.ravel()

        # Half-kick
        py_update_v(MM, Nx, Ny, Nz, aa, 0.5*delta_aa_step,
                    vomegam, Hf(aa), LL_box,
                    rra_flat, vva_flat, phi_old_flat)

        # Drift
        dt_drift = delta_aa_step / (a_mid**2 * Hf(a_mid))
        py_update_x(MM, Nx, Ny, Nz, a_mid, delta_aa_step, dt_drift,
                    rra_flat, vva_flat)

        # CRITICAL: robust periodic wrap after every drift
        periodic_wrap(rra_flat, Nx, Ny, Nz, MM)

        # Force at new positions
        _, phi_new   = compute_phi(rra_flat, MM, Nx, Ny, Nz,
                                   rho_b_inv, LL_box, vol)
        


        phi_new_flat = phi_new.ravel()

        # Half-kick
        py_update_v(MM, Nx, Ny, Nz, a_end, 0.5*delta_aa_step,
                    vomegam, Hf(a_end), LL_box,
                    rra_flat, vva_flat, phi_new_flat)

        aa = a_end

    # ---- Snapshot ----
    ro_ps      = cic_py(rra_flat.reshape(MM, 3), Nx, Ny, Nz, rho_b_inv)
    delta_grid = ro_ps / ro_ps.mean() - 1.0

    delta_proj       = np.sum(delta_grid, axis=2) / Nz
    delta_slices[jj] = delta_proj

    delta_k_ps        = fft3d_r2c_py(delta_grid * LL_box**3)
    delta_k_ps_deconv = cic_deconvolution(delta_k_ps, Nx, Ny, Nz)
    Pk_ps, _, _       = calpow(delta_k_ps_deconv, Nbin,
                                2*np.pi/LL_box, vol)

    Pk_history[jj] = Pk_ps
    a_history[jj]  = aa

    print(f"Snapshot {jj:2d}  z={nz[jj]:4f}  a={aa:.5f}  "
          f"max delta_proj={delta_proj.max():.4f}")

    # Safe wrap before writing
    periodic_wrap(rra_flat, Nx, Ny, Nz, MM)

    write_output(
        nbody_filename(nz[jj]),
        seed=SEED, output_flag=1,
        rra=rra_flat.reshape(MM, 3),
        vva=vva_flat.reshape(MM, 3),
        vaa=aa
    )
    check_positions(nbody_filename(nz[jj]), MM, Nx, label=f"z={nz[jj]}")

    # Check growth factor ratio
    D_ratio = Df(aa) / Df(a_init)
    Pk_expected_ratio = D_ratio**2
    Pk_measured_ratio = Pk_history[jj][3] / Pk_history[0][3]  # mid-k bin
    #print(f"z={nz[jj]}: D^2 ratio={Pk_expected_ratio:.4f}, "
    #  f"measured Pk ratio={Pk_measured_ratio:.4f}, "
    #  f"discrepancy={Pk_measured_ratio/Pk_expected_ratio:.4f}")

print("=== Simulation complete ===")
print(f"Output files: {[nbody_filename(z) for z in nz]}")

#############################################################
# pick a low‑k linear‑regime mode

k_linear_target = 0.1  # Mpc⁻¹
dist = np.abs(kmode_binned_ini - k_linear_target)
idx = np.argmin(dist)

print(" ")
print("Low‑k linear‑regime check (k ≈ 0.1 Mpc⁻¹):")
for jj in range(len(nz)):
    k_now  = kmode_binned_ini[idx]
    Pk_lin = Pk_history[0][idx] * (Df(a_history[jj])/Df(a_init))**2
    Pk_sim = Pk_history[jj][idx]
    ratio  = Pk_sim / Pk_lin
    print(f"  z={nz[jj]}  "
          f"k={k_now:.2f}  Pk_lin_Norm={Pk_lin:.2e}  Pk_sim={Pk_sim:.2e}  ratio={ratio:.2f}")

# =====================================================
# HALO CATALOGUE READER
# Reads halo_catalogue_<z> written by fof_main.c
# output_flag=1: positions in grid units -> convert to Mpc
# =====================================================
def read_halo_catalogue(fname, LL_box=0.14):
    with open(fname, 'rb') as f:
        f.read(4); f.read(256); f.read(4)   # header block
        f.read(4)
        Nclusters = struct.unpack('<i', f.read(4))[0]  # long=4B Windows
        f.read(4)
        f.read(4)
        data = np.frombuffer(
            f.read(Nclusters * 7 * 4), dtype=np.float32
        ).reshape(Nclusters, 7)
        f.read(4)

    mass = data[:, 0]              # 10^10 M_sun/h
    pos  = data[:, 1:4] * LL_box  # grid units -> Mpc
    vel  = data[:, 4:7]

    return mass, pos, vel

# =====================================================
# PLOTS
# =====================================================
plt.figure(figsize=(8, 6))
for jj in range(len(nz)):
    plt.loglog(kmode_binned_ini, Pk_history[jj],
               label=f"a={a_history[jj]:.4f} (z={nz[jj]})")
plt.xlabel(r"$k$ [Mpc$^{-1}$]")
plt.ylabel(r"$P(k)$ [Mpc$^3$]")
plt.title("Power Spectrum Evolution (CIC deconvolved)")
plt.legend(fontsize=7)
plt.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.savefig("plots/pk_evolution.png", dpi=150)
plt.show()

f_Pk_full         = interp1d(k_linear, pk_linear, kind='cubic',
                              fill_value="extrapolate")
Pk_full_at_binned = f_Pk_full(kmode_binned_ini)
Pk_binned_ini_raw, _, _ = calpow(delta_k_ini, Nbin, 2*np.pi/LL_box, vol)
Pk_binned_matched = Pk_binned_ini_raw * (
    Pk_full_at_binned / (Pk_binned_ini_raw + 1e-30)
)

plt.figure(figsize=(10, 7))
plt.loglog(k_linear, pk_linear, ls='-', color='k', alpha=0.8,
           label=r"Input linear $P(k)$ (z=199)")
plt.loglog(kmode_binned_ini, Pk_binned_matched, ls='', marker='s',
           ms=5, color='red', label="Initial binned $P(k)$")
colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(nz)))
for jj in range(len(nz)):
    plt.loglog(kmode_binned_ini, Pk_history[jj], ls='', marker='o',
               ms=4, alpha=0.8, color=colors[jj], label=rf"z={nz[jj]}")
plt.xlabel(r"$k$ [Mpc$^{-1}$]")
plt.ylabel(r"$P(k)$ [Mpc$^3$]")
plt.title("Full P(k) Evolution")
plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=7)
plt.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.savefig("plots/pk_full.png", dpi=150)
plt.show()

ncols = (len(nz) + 1) // 2
fig, axes = plt.subplots(2, ncols, figsize=(18, 8))
axes = axes.flatten()
for jj in range(len(nz)):
    im = axes[jj].imshow(delta_slices[jj].T, origin='lower', cmap='plasma')
    axes[jj].set_title(f"a={a_history[jj]:.4f}  (z={nz[jj]})")
    fig.colorbar(im, ax=axes[jj], shrink=0.7, label=r"$\delta$")
for ax in axes[len(nz):]:
    ax.axis('off')
plt.tight_layout()
plt.savefig("plots/density_slices.png", dpi=150)
plt.show()

rra_final = rra_flat.reshape(MM, 3)
plt.figure(figsize=(6, 6))
plt.scatter(rra_final[:, 0], rra_final[:, 1], s=0.3, alpha=0.3)
plt.xlim(0, Nx); plt.ylim(0, Ny)
plt.xlabel("x [grid units]")
plt.ylabel("y [grid units]")
plt.title(f"Final Particle Distribution (z=0), "
          f"Box={Nx*LL_box:.1f} Mpc")
plt.gca().set_aspect('equal')
plt.tight_layout()
plt.savefig("plots/particles_2d.png", dpi=150)
plt.show()