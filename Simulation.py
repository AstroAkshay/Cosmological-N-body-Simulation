import os
import numpy as np
import inspect
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import time


# =========================================
# Enable OpenMP threads
# =========================================
os.environ["OMP_NUM_THREADS"] = "8"   # change to number of CPU cores

import sys, os
print("Current dir:", os.getcwd())
print("sys.path:", sys.path)
print("nbodysim exists?", os.path.exists('nbodysim'))

# Add parent directory to Python path
import_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, import_path)
print("Updated sys.path:", sys.path)


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

total_start = time.perf_counter()

# =====================================================
# 1. Simulation parameters
# =====================================================
par = PyParams(N1=64, N2=64, N3=64, Nbin=10, LL=0.07, NF=1, vaa = 0.047)

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
# 2. Load input power spectrum
# =====================================================
def load_k_Pk():

    module_dir = os.path.dirname(inspect.getfile(load_k_Pk))

    df = pd.read_csv(
        os.path.join(module_dir, "data", "P_CDM(20).csv"),
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

print("\nDensity field stats :", "\nMin. ",delta_x.min(),"\nMax. ", delta_x.max(),"\nMean ", delta_x.mean())


# =====================================================
# 4. Zel'dovich initial conditions
# =====================================================
Cx, Cy, Cz = 2*np.pi/Nx, 2*np.pi/Ny, 2*np.pi/Nz

disp_kx = np.zeros_like(delta_k)
disp_ky = np.zeros_like(delta_k)
disp_kz = np.zeros_like(delta_k)

grad_phi_py(0, delta_k, disp_kx, Nx, Ny, Nz, Cx, Cy, Cz, vol)
grad_phi_py(1, delta_k, disp_ky, Nx, Ny, Nz, Cx, Cy, Cz, vol)
grad_phi_py(2, delta_k, disp_kz, Nx, Ny, Nz, Cx, Cy, Cz, vol)

disp_x = fft3d_c2r_py(disp_kx, Nx, Ny, Nz).ravel()
disp_y = fft3d_c2r_py(disp_ky, Nx, Ny, Nz).ravel()
disp_z = fft3d_c2r_py(disp_kz, Nx, Ny, Nz).ravel()

a_init = float(par.vaa)

H_init = 100 * vhh * np.sqrt(vomegam / a_init**3 + vomegalam)

f_init = (vomegam / a_init**3 /
          (vomegam / a_init**3 + vomegalam))**0.55

vfac = a_init * a_init * H_init * f_init


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

phi = np.zeros((Nx, Ny, Nz), dtype=np.float64)


# =====================================================
# 5. INITIAL POWER SPECTRUM
# =====================================================
rho_b_inv = float((Nx*Ny*Nz) / Npoints)

ro_ini = cic_py(rra, Nx, Ny, Nz, rho_b_inv)

delta_ini = ro_ini / ro_ini.mean() - 1.0

delta_k_ini = fft3d_r2c_py(delta_ini * LL**3)

Pk_ini, kmode_ref, _ = calpow(
    delta_k_ini,
    Nbin,
    2*np.pi / LL,
    vol
)


# =====================================================
# 6. Time evolution
# =====================================================
a_start = a_init
a_end = 1.0
da = par.delta_aa

nsteps = int((a_end - a_start)/da) + 1

a_values = np.linspace(a_start, a_end, nsteps)

Pk_history = np.zeros((nsteps, Nbin))
Pk_history[0] = Pk_ini

a_history = a_values.copy()

print("\nStarting evolution...")


# Preallocate arrays
ro = np.zeros((Nx, Ny, Nz))
delta_x_grid = np.zeros_like(ro)

delta_k_grid = np.zeros((Nx, Ny, Nz//2 + 1), dtype=np.complex128)

phi_k = np.zeros_like(delta_k_grid)


for step, a in enumerate(a_values[1:], 1):

    a_float = float(a)

    inv_a3 = 1.0 / a_float**3

    H_float = 100.0 * vhh * np.sqrt(
        vomegam * inv_a3 + vomegalam
    )

    coeff_x = da * inv_a3 / H_float

    print(f"--- Step {step}/{nsteps-1}, a = {a_float:.4f} ---")

    ro[:] = cic_py(rra, Nx, Ny, Nz, rho_b_inv)

    delta_x_grid[:] = ro / ro.mean() - 1.0

    delta_k_grid[:] = fft3d_r2c_py(delta_x_grid * LL**3)

    phi_k[:] = apply_inverse_laplacian(delta_k_grid, LL, vol)

    phi[:] = fft3d_c2r_py(phi_k / vol, Nx, Ny, Nz)

    py_update_v(
        a_float, da,
        rra, vva,
        Nx, Ny, Nz,
        vomegam, LL,
        H_float,
        phi
    )

    py_update_x(
        a_float, da,
        rra, vva,
        Nx, Ny, Nz,
        coeff_x
    )

    ro_ps = cic_py(rra, Nx, Ny, Nz, rho_b_inv)

    delta_ps = ro_ps / ro_ps.mean() - 1.0

    delta_k_ps = fft3d_r2c_py(delta_ps * LL**3)

    Pk_ps, _, _ = calpow(
        delta_k_ps,
        Nbin,
        2*np.pi / LL,
        vol
    )

    Pk_history[step] = Pk_ps


print("\nEvolution completed")

total_end = time.perf_counter()

total_runtime = total_end - total_start

print(f"\nTotal runtime of evolution : {total_runtime:.3f} seconds")


# =========================================
# Final density field
# =========================================
ro_final = cic_py(rra, Nx, Ny, Nz, rho_b_inv)

delta_final = ro_final / ro_final.mean() - 1.0

delta_final_3d = delta_final.reshape(Nx, Ny, Nz)
delta_ini_3d = delta_ini.reshape(Nx, Ny, Nz)

# =====================================================
# 7. Plot evolution
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

plt.xlabel("k (Mpc⁻¹)")
plt.ylabel("P(k) (Mpc³)")
plt.legend()

plt.title("Power Spectrum Evolution")

plt.show()


################### INITIAL VS FINAL DENSITY FIELD PLOT ####################################3

from mpl_toolkits.axes_grid1 import make_axes_locatable

slice_z = Nz // 2

initial_slice = delta_ini_3d[:, :, slice_z]
final_slice = delta_final_3d[:, :, slice_z]

vmin = min(initial_slice.min(), final_slice.min())
vmax = max(initial_slice.max(), final_slice.max())

fig, axes = plt.subplots(1, 2, figsize=(10,4))

im0 = axes[0].imshow(
    initial_slice,
    origin="lower",
    cmap="plasma",
    vmin=vmin,
    vmax=vmax
)

axes[0].set_title("Initial Density Field")

im1 = axes[1].imshow(
    final_slice,
    origin="lower",
    cmap="plasma",
    vmin=vmin,
    vmax=vmax
)

axes[1].set_title("Final Density Field")

# create colorbar axis
divider = make_axes_locatable(axes[1])
cax = divider.append_axes("right", size="5%", pad=0.1)

cbar = fig.colorbar(im1, cax=cax)
cbar.set_label(r"$\delta(x)$")

plt.tight_layout()
plt.show()

