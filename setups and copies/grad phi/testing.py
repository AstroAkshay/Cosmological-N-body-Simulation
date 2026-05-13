import numpy as np
import gradphi  # your compiled module

# Example dimensions
N1, N2, N3 = 128, 128, 128
Cx, Cy, Cz = 1.0, 1.0, 1.0
vol = 1.0

# Your existing delta_k array (complex64)
delta_k = np.random.randn(N1, N2, N3//2+1).astype(np.complex64)

# Allocate FFTW-compatible arrays
ro_fftw = np.empty((N1, N2, N3//2+1, 2), dtype=np.float32)
va_fftw = np.zeros_like(ro_fftw)

# Fill ro_fftw with real and imaginary parts
ro_fftw[..., 0] = delta_k.real
ro_fftw[..., 1] = delta_k.imag

# Compute x-component (ix=0)
gradphi.grad_phi_py(0, ro_fftw, va_fftw, N1, N2, N3, Cx, Cy, Cz, vol)

# Compute y-component (ix=1)
gradphi.grad_phi_py(1, ro_fftw, va_fftw, N1, N2, N3, Cx, Cy, Cz, vol)

# Compute z-component (ix=2)
gradphi.grad_phi_py(2, ro_fftw, va_fftw, N1, N2, N3, Cx, Cy, Cz, vol)
