from code.fftgrid_wrapper import generate_delta_k_3d_py

Nx, Ny, Nz = 2, 2, 2
delta_k = generate_delta_k_3d_py(Nx, Ny, Nz)

print(delta_k.shape)   # (2, 2, 2)
print(delta_k)
