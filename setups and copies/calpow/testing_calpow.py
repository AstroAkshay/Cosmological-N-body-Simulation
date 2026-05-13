import numpy as np
from code.calpow_module import calpow

# Example grid size
N1, N2, N3 = 64, 64, 64
Nbin = 20
tpibyL = 2.0 * np.pi / 100.0
vol = 100.0 ** 3

# Create a fake delta_k (half-complex shape like FFTW)
delta_k = np.random.randn(N1, N2, N3//2 + 1) + 1j * np.random.randn(N1, N2, N3//2 + 1)
delta_k = delta_k.astype(np.complex64)

# Run power spectrum calculation
power, kmode, no = calpow(delta_k, Nbin, tpibyL, vol)

print("Power spectrum:", power)
print("k bins:", kmode)
print("Counts:", no)
