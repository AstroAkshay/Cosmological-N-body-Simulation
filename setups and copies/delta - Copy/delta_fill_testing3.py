import numpy as np
from code.delta_fill_helper import delta_fill


def Pk(kx, ky, kz):
    k = np.sqrt(kx**2 + ky**2 + kz**2)
    return np.exp(-k**2)  # example spectrum

N = 4
A = np.zeros((N, N, N//2 + 1, 2), dtype=np.float32)
delta_fill(A, N, N, N, Pk)
print("A :\n",A)

delta_k = A[..., 0] + 1j * A[..., 1]
print("DELTA K :\n",delta_k)

delta_x = np.fft.irfftn(delta_k, s=(N, N, N))
print("DELTA X :\n",delta_x)

print("δ(k) shape:", delta_k.shape)
print("Max Im[δ(x)] =", np.abs(delta_x.imag).max())
