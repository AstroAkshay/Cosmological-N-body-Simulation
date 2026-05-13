# cython: language_level=3
import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free

# Declare the C function
cdef extern from "fftgrid.h":
    void generate_delta_k_3d(size_t Nx, size_t Ny, size_t Nz, float complex* delta_k_3d)

def generate_delta_k_3d_py(int Nx, int Ny, int Nz):
    """
    Python wrapper for the C function
    """
    cdef int total_points = Nx * Ny * Nz
    cdef np.ndarray[np.complex64_t, ndim=1] delta_k_1d = np.zeros(total_points, dtype=np.complex64)
    generate_delta_k_3d(Nx, Ny, Nz, <float complex*>delta_k_1d.data)
    return delta_k_1d.reshape((Nx, Ny, Nz))
