import os
import numpy as np
import inspect
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

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
# 1. Simulation parameters
# =====================================================
par = PyParams(N1=256,N2=256,N3=256,Nbin=10,LL=0.07)

Nx, Ny, Nz = int(par.N1), int(par.N2), int(par.N3)
LL = float(par.LL)
vhh = float(par.vhh)
vomegam = float(par.vomegam)
vomegalam = float(par.vomegalam)
Nbin = par.Nbin
vol = Nx * Ny * Nz * LL**3
NF = par.NF
Npoints = (Nx // NF) * (Ny // NF) * (Nz // NF)

print("Parameters for cosmological evolution :\n")
par.print()

# =====================================================
# 2. Load input P(k)
# =====================================================
def load_k_Pk():
    module_dir = os.path.dirname(inspect.getfile(load_k_Pk))
    df = pd.read_csv(
        os.path.join(module_dir, "data", "P_DAO.csv"),
        header=None,
        names=["k", "Pk"]
    )
    return df["k"].values, df["Pk"].values

k, pk = load_k_Pk()

Pk_spline = CubicSpline(k, pk, bc_type="natural")

# =====================================================
# 3. Generate Hermitian delta(k)
# =====================================================
delta_k_raw = generate_hermitian_grid(
    Nx, Ny, Nz, Nx * LL, Pk_spline
)

delta_k_raw *= np.sqrt(vol)
delta_k_raw[0, 0, 0] = 0.0 + 0.0j

delta_k = np.zeros((Nx, Ny, Nz//2 + 1), dtype=np.complex128)
delta_k[:] = delta_k_raw[:, :, :Nz//2 + 1]

delta_x = fft3d_c2r_py(delta_k / vol, Nx, Ny, Nz)
#print("δ(x): mean =", delta_x.mean(),
#      "min =", delta_x.min(),
#      "max =", delta_x.max())

# =====================================================
# 4. Zel'dovich initial conditions
# =====================================================
Cx, Cy, Cz = 2*np.pi/Nx, 2*np.pi/Ny, 2*np.pi/Nz
va_fftw = np.zeros_like(delta_k)

disp_k = [
    grad_phi_py(
        i, delta_k, va_fftw,
        Nx, Ny, Nz,
        Cx, Cy, Cz,
        vol
    ).copy()
    for i in range(3)
]

disp_x = fft3d_c2r_py(disp_k[0], Nx, Ny, Nz).ravel()
disp_y = fft3d_c2r_py(disp_k[1], Nx, Ny, Nz).ravel()
disp_z = fft3d_c2r_py(disp_k[2], Nx, Ny, Nz).ravel()

a_init = float(par.vaa)
H_init = 100 * vhh * np.sqrt(vomegam / a_init**3 + vomegalam)
f_init = (vomegam / a_init**3 /
          (vomegam / a_init**3 + vomegalam))**0.55

vfac = a_init * a_init * H_init * f_init
#print("vfac =", vfac)

rra = np.zeros((Npoints, 3), dtype=np.float64)
vva = np.zeros((Npoints, 3), dtype=np.float64)

py_Zel_move_gradphi(
    float(vfac),
    rra.ravel(),
    vva.ravel(),
    disp_x, disp_y, disp_z,
    Nx, Ny, Nz, NF, LL
)

rra %= np.array([Nx, Ny, Nz], dtype=np.float64)
rra = np.ascontiguousarray(rra)
vva = np.ascontiguousarray(vva)

phi = np.zeros((Nx, Ny, Nz), dtype=np.float64)
phi = np.ascontiguousarray(phi)

# =====================================================
# 5. INITIAL POWER SPECTRUM (a = 0.008)
# =====================================================
rho_b_inv = float((Nx*Ny*Nz) / Npoints)

ro_ini = cic_py(rra, Nx, Ny, Nz, rho_b_inv)  # INITAL DENSITY FIELD AFTER ZEL DOVICH DISPLACEMENT 
delta_ini = ro_ini / ro_ini.mean() - 1.0

delta_k_ini = fft3d_r2c_py(delta_ini * LL**3)
Pk_ini, kmode_ref, _ = calpow(
    delta_k_ini,
    Nbin,
    2*np.pi / LL,
    vol
)

Pk_history = [Pk_ini.copy()]
a_history = [a_init]

#print("Initial P(k) measured at a =", a_init)

# =====================================================
# 6. Time evolution
# =====================================================
a_start = a_init
a_end = 1.0
da = par.delta_aa

nsteps = int((a_end - a_start)/da) + 1
a_values = np.linspace(a_start, a_end, nsteps)

print("\nStarting evolution...")

for step, a in enumerate(a_values[1:], 1):

    a_float = float(a)
    da_float = float(da)

    H_float = 100.0 * vhh * np.sqrt(
        vomegam / a_float**3 + vomegalam
    )

    coeff_x = da_float / (a_float**3 * H_float)

    print(f"--- Step {step}/{nsteps-1}, a = {a_float:.4f} ---")

    # --- Density → potential ---
    ro = cic_py(rra, Nx, Ny, Nz, rho_b_inv)
    delta_x_grid = ro / ro.mean() - 1.0

    delta_k_grid = fft3d_r2c_py(delta_x_grid * LL**3)
    phi_k = apply_inverse_laplacian(delta_k_grid, LL, vol)

    phi[:] = fft3d_c2r_py(phi_k / vol, Nx, Ny, Nz)

    # --- Velocity update ---
    py_update_v(
        a_float, da_float,
        rra, vva,
        Nx, Ny, Nz,
        vomegam, LL,
        H_float,
        phi
    )

    # --- Position update ---
    py_update_x(
        a_float, da_float,
        rra, vva,
        Nx, Ny, Nz,
        coeff_x
    )

    # --- RMS velocity ---
    v_rms = np.sqrt(np.mean(vva**2))
    #print(f"v_rms = {v_rms:.6e}")

    # --- Power spectrum ---
    ro_ps = cic_py(rra, Nx, Ny, Nz, rho_b_inv)
    delta_ps = ro_ps / ro_ps.mean() - 1.0

    delta_k_ps = fft3d_r2c_py(delta_ps * LL**3)
    Pk_ps, _, _ = calpow(
        delta_k_ps,
        Nbin,
        2*np.pi / LL,
        vol
    )

    Pk_history.append(Pk_ps.copy())
    a_history.append(a_float)

print("\nEvolution completed")

# =====================================================
# 7. Convert histories
# =====================================================
Pk_history = np.array(Pk_history)
a_history = np.array(a_history)

# =====================================================
# 8. Plot evolution INCLUDING initial spectrum
# =====================================================
plt.figure()

for i in range(0, len(a_history), max(1, len(a_history)//6)):
    plt.loglog(
        kmode_ref,
        Pk_history[i],
        label=f"a={a_history[i]:.3f}"
    )

plt.loglog(
    kmode_ref,
    Pk_spline(kmode_ref),
    "k--",
    label="Input"
)

plt.xlabel("k (MPc^-1)")
plt.ylabel("P(k) (Mpc^3)")
plt.legend()
plt.title("Power Spectrum Evolution")
plt.show()

# =====================================================
# 9. Large-scale growth check
# =====================================================
k_index = np.argmin(np.abs(kmode_ref - 0.2))
Pk_lowk = Pk_history[:, k_index]

plt.plot(a_history, Pk_lowk / Pk_lowk[0])
plt.xlabel("a")
plt.ylabel("P(k)/P(k_initial)")
plt.title("Linear-regime growth")
plt.grid()
plt.show()

# =====================================================
# 10. Plot full input spectrum + measured bins
# =====================================================

plt.figure()

# --- Plot continuous input spectrum ---
k_dense = np.logspace(np.log10(k.min()), np.log10(k.max()), 1000)
plt.loglog(
    k_dense,
    Pk_spline(k_dense),
    'k-',
    label="Input P(k)"
)

# --- Plot measured spectrum at selected times ---
for i in range(0, len(a_history), max(1, len(a_history)//6)):
    plt.loglog(
        kmode_ref,
        Pk_history[i],
        'o',
        markersize=4,
        label=f"Measured (a={a_history[i]:.3f})"
    )

plt.xlabel("k (Mpc^-1)")
plt.ylabel("P(k) (Mpc^3)")
plt.legend()
plt.title("Full Power Spectrum with Binned Measurement")
plt.grid(True, which="both")
plt.show()

# =====================================================
# 11. Plot Initial and Final Density Fields
# =====================================================

# Reshape to 3D grids
delta_ini_3d = delta_ini.reshape(Nx, Ny, Nz)
delta_final_3d = delta_ps.reshape(Nx, Ny, Nz)

# Take mid-plane slice
z_slice = Nz // 2

plt.figure(figsize=(12, 5))

# --- Initial ---
# --- Select slice ---
z_slice = Nz // 2   # or any slice index you want

# --- Extract slices ---
delta_ini_slice   = delta_ini_3d[:, :, z_slice]
delta_final_slice = delta_final_3d[:, :, z_slice]

# --- Global color limits (shared scaling) ---
vmin = min(delta_ini_slice.min(), delta_final_slice.min())
vmax = max(delta_ini_slice.max(), delta_final_slice.max())

# --- Create figure ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

# --- Initial field ---
im0 = axes[0].imshow(
    delta_ini_slice,
    origin='lower',
    cmap='plasma',
    extent=[0, Nx*LL, 0, Ny*LL],
    vmin=vmin,
    vmax=vmax,
    aspect='equal'
)
axes[0].set_title(f"Initial Density Field (a = {a_history[0]:.3f})")
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")

# --- Final field ---
im1 = axes[1].imshow(
    delta_final_slice,
    origin='lower',
    cmap='plasma',
    extent=[0, Nx*LL, 0, Ny*LL],
    vmin=vmin,
    vmax=vmax,
    aspect='equal'
)
axes[1].set_title(f"Final Density Field (a = {a_history[-1]:.3f})")
axes[1].set_xlabel("x")
axes[1].set_ylabel("y")

# --- Shared colorbar ---
cbar = fig.colorbar(im1, ax=axes, shrink=0.9)
cbar.set_label("Density Contrast δ")

plt.show()

