# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
import numpy as np
cimport numpy as np

cdef extern from "hermitian_grid.h":
    ctypedef struct complex32:
        float real
        float imag

    void generate_hermitian_grid(size_t Nx, size_t Ny, size_t Nz, complex32* delta_k_3d)

def generate_hermitian_grid_py(int Nx, int Ny, int Nz):
    """
    High-performance Python wrapper for Hermitian grid generator.
    Returns np.ndarray (Nx, Ny, Nz), dtype=np.complex64
    """
    cdef int total_points = Nx * Ny * Nz
    # Allocate numpy array for complex64 (same layout as complex32[2 floats])
    cdef np.ndarray[np.complex64_t, ndim=1] delta_k_flat = np.empty(total_points, dtype=np.complex64)

    # Direct pointer to NumPy data as complex32
    generate_hermitian_grid(Nx, Ny, Nz, <complex32*> delta_k_flat.data)

    return delta_k_flat.reshape((Nx, Ny, Nz))
