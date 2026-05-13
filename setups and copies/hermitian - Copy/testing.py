from code.hermitian_wrapper import generate_hermitian_grid_py
import numpy as np

Nx, Ny, Nz = 4,4,4
delta_k = generate_hermitian_grid_py(Nx, Ny, Nz)
print(delta_k)

print("Shape:", delta_k.shape)
print("Dtype:", delta_k.dtype)
print("Sample slice (real part):\n", np.real(delta_k[:, :, 0]))
print("Sample slice (imag part):\n", np.imag(delta_k[:, :, 0]))
