import numpy as np

# Original grid
Nx, Ny, Nz = 2, 2, 2
total_points = Nx*Ny*Nz

# Number of independent modes (half)
n_half = total_points // 2  # 4

# Planar array for independent modes: [real, imag]
rho_half = np.zeros((n_half, 2), dtype=np.float32)

# Fill with Gaussian numbers
randn = np.random.standard_normal(rho_half.size).astype(np.float32)
rho_half[:] = randn.reshape(rho_half.shape)
print("rho_half (independent modes):\n", rho_half)

# Create full array by appending complex conjugates
# Convert to complex numbers first
delta_half = rho_half[:,0] + 1j*rho_half[:,1]

# Full delta_k array
delta_full = np.zeros(total_points, dtype=np.complex64)
delta_full[:n_half] = delta_half
delta_full[n_half:] = np.conj(delta_half)  # Hermitian symmetry

# Reshape to 3D
delta_k_3d = delta_full.reshape(Nx, Ny, Nz)
print("delta_k_3d shape:", delta_k_3d.shape)
print(delta_k_3d)
