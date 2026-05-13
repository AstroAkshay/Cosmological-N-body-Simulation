import numpy as np
from code.hermitian_wrapper import generate_hermitian_grid

# Define grid size
Nx, Ny, Nz = 4,4,4
boxlen = 0.07

# Example Python power spectrum function
def Pk(k):
    return 1.0 / (1.0 + k**2)  # just a simple test spectrum

# Generate Hermitian delta_k field
delta_k = generate_hermitian_grid(Nx, Ny, Nz, boxlen, Pk)

print("delta_k shape:", delta_k.shape,delta_k)

#print("Sample values (real part):", delta_k.real.flatten()[:10])
#print("Sample values (imag part):", delta_k.imag.flatten()[:10])

print(np.argwhere(delta_k == 0+0j))