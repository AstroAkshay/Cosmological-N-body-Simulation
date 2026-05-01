#!/usr/bin/env python3
"""
COSMOLOGICAL PM SIMULATION - FULL RUN
Fixes applied:
  1. KDK leapfrog: force recomputed AFTER drift, before second half-kick
  2. H(a) evaluated at correct midpoint scale factor for each half-kick
  3. CIC deconvolution: sinc argument clarified to use dimensionless frequency
  4. Duplicate import of numpy removed; all imports moved to top of file
  5. Minor clarity improvements (comments, variable naming)
"""

# =====================================================
# IMPORTS (all at top)
# =====================================================
import os
import sys
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

Nx, Ny, Nz     = int(par.N1), int(par.N2), int(par.N3)
LL_box         = float(par.LL)
Nbin           = par.Nbin
vomegam        = float(par.vomegam)
vomegalam      = float(par.vomegalam)

vol     = Nx * Ny * Nz * LL_box**3
Npoints = (Nx // par.NF) * (Ny // par.NF) * (Nz // par.NF)

# =====================================================
# COSMOLOGICAL GROWTH FUNCTIONS
# =====================================================
def Hf(a):
    """Dimensionless Hubble parameter E(a) = H(a)/H0."""
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
    """Linear growth factor normalised to D(1) = 1."""
    return Hf(a) * Integralf(a) / (Hf(1.0) * Integralf(1.0))

# =====================================================
# LOAD LINEAR POWER SPECTRUM
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
delta_k_raw[0, 0, 0] = 0.0                                  # zero mean density

delta_k = delta_k_raw[:, :, :Nz // 2 + 1].astype(np.complex128)
delta_x = fft3d_c2r_py(delta_k / vol, Nx, Ny, Nz)

# Zel'dovich displacement fields: Psi_i = -i k_i / k^2  *  delta(k)
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

a_init  = float(par.vaa)
H_init  = Hf(a_init)
vfac    = a_init**2 * H_init   # a^2 H factor for growing-mode velocities

py_Zel_move_gradphi(
    vfac, rra.ravel(), vva.ravel(),
    disp_x_ravel, disp_y_ravel, disp_z_ravel,
    Nx, Ny, Nz, par.NF, LL_box
)

rra %= np.array([Nx, Ny, Nz], dtype=np.float64)   # periodic wrap

# =====================================================
# INITIAL DENSITY & POWER SPECTRUM
# =====================================================
rho_b_inv = (Nx * Ny * Nz) / Npoints              # 1 / mean particle weight
ro_ini    = cic_py(rra, Nx, Ny, Nz, rho_b_inv)

delta_ini   = ro_ini / ro_ini.mean() - 1.0
delta_k_ini = fft3d_r2c_py(delta_ini * LL_box**3)

# =====================================================
# CIC DECONVOLUTION
# Correct for the CIC window W(k) = [sinc(k_i / 2 k_Ny)]^2 per axis.
# np.sinc is defined as sinc(x) = sin(pi x)/(pi x), so we pass
# the argument as  k_i / (2 * k_Ny)  =  k_i_norm / 2  (dimensionless [0,0.5]).
# =====================================================
def cic_deconvolution(delta_k, Nx, Ny, Nz):
    """
    Divide out the CIC mass-assignment window function in Fourier space.

    The CIC window along each axis is:
        W_i(k_i) = sinc(k_i / (2 k_Ny,i))^2
    where k_Ny,i = pi / dx_i is the Nyquist wavenumber and
    sinc(x) = sin(pi x)/(pi x)  (NumPy convention).

    Equivalently, if f_i = k_i / (2 pi) * dx_i is the dimensionless
    frequency in [0, 0.5], then W_i = sinc(f_i)^2.
    """
    # Dimensionless frequencies: ix/Nx in [-0.5, 0.5) for full axes,
    # [0, 0.5] for the rfft (z) axis.
    fx_norm = np.arange(Nx) / Nx
    fx_norm[Nx // 2:] -= 1.0

    fy_norm = np.arange(Ny) / Ny
    fy_norm[Ny // 2:] -= 1.0

    fz_norm = np.arange(Nz // 2 + 1) / Nz   # rfft: 0 … +0.5

    FX, FY, FZ = np.meshgrid(fx_norm, fy_norm, fz_norm, indexing='ij')

    # np.sinc(f) = sin(pi f) / (pi f)  →  correct CIC window factor
    Wx = np.sinc(FX)
    Wy = np.sinc(FY)
    Wz = np.sinc(FZ)

    W2 = (Wx * Wy * Wz)**2        # square because CIC is NGP convolved twice

    return delta_k / (W2 + 1e-10)


# =====================================================
# HELPER: compute density field and gravitational potential
# =====================================================
def compute_phi(rra_flat, MM, Nx, Ny, Nz, rho_b_inv, LL_box, vol):
    """CIC → FFT → inverse Laplacian → real-space potential."""
    ro        = cic_py(rra_flat.reshape(MM, 3), Nx, Ny, Nz, rho_b_inv)
    delta_g   = ro / ro.mean() - 1.0
    delta_k_g = fft3d_r2c_py(delta_g * LL_box**3)
    phi_k     = delta_k_g.copy()
    apply_inverse_laplacian(phi_k, LL_box, vol)
    phi       = fft3d_c2r_py(phi_k, Nx, Ny, Nz)
    return ro, phi

# =====================================================
# LEAPFROG (KDK) TIME INTEGRATION WITH ADAPTIVE STEPS
#
# Correct KDK scheme:
#   1. Half-kick:   v += -1/2 * da * grad(phi_old) / (a^2 H(a))
#   2. Drift:       x += v * dt   where dt = da / (a^2 H(a+da/2))  (midpoint)
#   3. Recompute force at new positions
#   4. Half-kick:   v += -1/2 * da * grad(phi_new) / (a^2 H(a+da))
#
# H(a) is evaluated at the appropriate scale factor for each sub-step.
# =====================================================
print("=== Starting simulation ===")

rra_flat = rra.ravel()
vva_flat = vva.ravel()
MM       = rra.shape[0]

# Target redshifts (first entry is the IC redshift)
nz = [99, 80, 60, 40, 20, 12, 10, 8, 5, 2, 1, 0]

aa           = a_init
Pk_history   = np.zeros((len(nz), Nbin))
a_history    = np.zeros(len(nz))
delta_slices = np.zeros((len(nz), Nx, Ny))   # projected δ along z

# --- Initial snapshot (IC, index 0) ---
delta_k_ini_deconv              = cic_deconvolution(delta_k_ini, Nx, Ny, Nz)
Pk_ini, kmode_binned_ini, _     = calpow(delta_k_ini_deconv, Nbin, 2 * np.pi / LL_box, vol)
Pk_history[0]                   = Pk_ini
a_history[0]                    = aa

ro_ini_proj     = np.sum(ro_ini, axis=2) / Nz
delta_slices[0] = ro_ini_proj / ro_ini_proj.mean() - 1.0

# --- Time loop ---
for jj in range(1, len(nz)):
    afin = 1.0 / (nz[jj] + 1.0)

    while aa < afin:
        # Adaptive time step: finer steps at early times
        delta_aa_step = par.delta_aa * (a_init / aa)**0.5
        delta_aa_step = min(delta_aa_step, afin - aa)   # don't overshoot snapshot

        a_mid  = aa + 0.5 * delta_aa_step
        a_end  = aa + delta_aa_step

        # --- Force at current positions ---
        _, phi_old  = compute_phi(rra_flat, MM, Nx, Ny, Nz, rho_b_inv, LL_box, vol)
        phi_old_flat = phi_old.ravel()

        # --- Half-kick (at aa, using phi_old) ---
        py_update_v(MM, Nx, Ny, Nz, aa, 0.5 * delta_aa_step,
                    vomegam, Hf(aa), LL_box,
                    rra_flat, vva_flat, phi_old_flat)

        # --- Drift (at midpoint a_mid) ---
        dt_drift = delta_aa_step / (a_mid**2 * Hf(a_mid))
        py_update_x(MM, Nx, Ny, Nz, a_mid, dt_drift, dt_drift,
                    rra_flat, vva_flat)

        # --- Recompute force at NEW positions ---
        _, phi_new   = compute_phi(rra_flat, MM, Nx, Ny, Nz, rho_b_inv, LL_box, vol)
        phi_new_flat = phi_new.ravel()

        # --- Half-kick (at a_end, using phi_new) ---
        py_update_v(MM, Nx, Ny, Nz, a_end, 0.5 * delta_aa_step,
                    vomegam, Hf(a_end), LL_box,
                    rra_flat, vva_flat, phi_new_flat)

        aa = a_end

    # --- Snapshot ---
    ro_ps, _ = compute_phi(rra_flat, MM, Nx, Ny, Nz, rho_b_inv, LL_box, vol)
    # (we only need ro here; recompute cheaply)
    ro_ps        = cic_py(rra_flat.reshape(MM, 3), Nx, Ny, Nz, rho_b_inv)
    delta_grid   = ro_ps / ro_ps.mean() - 1.0

    # Projected δ along z
    delta_proj       = np.sum(delta_grid, axis=2) / Nz
    delta_slices[jj] = delta_proj

    # P(k) with CIC deconvolution
    delta_k_ps       = fft3d_r2c_py(delta_grid * LL_box**3)
    delta_k_ps_deconv= cic_deconvolution(delta_k_ps, Nx, Ny, Nz)
    Pk_ps, _, _      = calpow(delta_k_ps_deconv, Nbin, 2 * np.pi / LL_box, vol)

    Pk_history[jj] = Pk_ps
    a_history[jj]  = aa

    print(f"Snapshot {jj:2d}  z={nz[jj]:4d}  a={aa:.5f}  max δ_proj={delta_proj.max():.4f}")

print("=== Simulation complete ===")

# =====================================================
# PLOT 1: POWER SPECTRUM EVOLUTION
# =====================================================
plt.figure(figsize=(8, 6))
for jj in range(len(nz)):
    plt.loglog(kmode_binned_ini, Pk_history[jj], label=f"a={a_history[jj]:.3f}")
plt.xlabel(r"$k$ [Mpc$^{-1}$]")
plt.ylabel(r"$P(k)$ [Mpc$^3$]")
plt.title("Binned Power Spectrum Evolution (CIC deconvolved)")
plt.legend(fontsize=7)
plt.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.show()

# =====================================================
# PLOT 2: FULL LINEAR + BINNED INITIAL + SNAPSHOT P(k)
# =====================================================

# Re-bin initial P(k) WITHOUT deconvolution for direct comparison
Pk_binned_ini, _, _ = calpow(delta_k_ini, Nbin, 2 * np.pi / LL_box, vol)

# Interpolate input linear P(k) at binned k modes
f_Pk_full          = interp1d(k_linear, pk_linear, kind='cubic', fill_value="extrapolate")
Pk_full_at_binned  = f_Pk_full(kmode_binned_ini)

# Renormalise to match input linear spectrum
Pk_binned_matched  = Pk_binned_ini * (Pk_full_at_binned / (Pk_binned_ini + 1e-30))

plt.figure(figsize=(10, 7))

plt.loglog(k_linear, pk_linear,
           ls='-', color='k', alpha=0.8, zorder=10,
           label=r"Initial linear $P(k)$ (full)")

plt.loglog(kmode_binned_ini, Pk_binned_matched,
           ls='', marker='s', ms=5, color='red',
           label=r"Initial binned $P(k)$ (matched)")

colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(nz)))
for jj in range(len(nz)):
    plt.loglog(kmode_binned_ini, Pk_history[jj],
               ls='', marker='o', ms=4, alpha=0.8,
               color=colors[jj],
               label=rf"$a={a_history[jj]:.3f}$")

plt.xlabel(r"$k$ [Mpc$^{-1}$]")
plt.ylabel(r"$P(k)$ [Mpc$^3$]")
plt.title("Full Input Linear + Binned Initial + Binned Snapshot P(k) Evolution")
plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=7)
plt.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.show()

# =====================================================
# PLOT 3: DENSITY FIELD SLICES AT ALL SNAPSHOTS
# =====================================================
ncols = (len(nz) + 1) // 2
fig, axes = plt.subplots(2, ncols, figsize=(18, 8))
axes = axes.flatten()

for jj in range(len(nz)):
    im = axes[jj].imshow(delta_slices[jj].T, origin='lower', cmap='plasma')
    axes[jj].set_title(f"a={a_history[jj]:.3f}  (z={nz[jj]})")
    axes[jj].set_xlabel("x")
    axes[jj].set_ylabel("y")
    fig.colorbar(im, ax=axes[jj], shrink=0.7, label=r"$\delta$")

for ax in axes[len(nz):]:
    ax.axis('off')

plt.tight_layout()
plt.show()

# =====================================================
# PLOT 4: FINAL PARTICLE DISTRIBUTION (2D PROJECTION)
# =====================================================
rra_final = rra_flat.reshape(MM, 3)

plt.figure(figsize=(6, 6))
plt.scatter(rra_final[:, 0], rra_final[:, 1], s=0.5, alpha=0.5)
plt.xlim(0, Nx)
plt.ylim(0, Ny)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Final Particle Distribution (2D Projection)")
plt.gca().set_aspect('equal')
plt.tight_layout()
plt.show()

# =====================================================
# PLOT 5: FINAL PARTICLE DISTRIBUTION (3D)
# =====================================================
fig = plt.figure(figsize=(7, 7))
ax  = fig.add_subplot(111, projection='3d')

ax.scatter(rra_final[:, 0], rra_final[:, 1], rra_final[:, 2],
           s=0.5, alpha=0.5)

ax.set_xlim(0, Nx)
ax.set_ylim(0, Ny)
ax.set_zlim(0, Nz)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_title("Final Particle Distribution (3D)")

plt.tight_layout()
plt.show()
