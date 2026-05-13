# cython: language_level=3
from libc.stddef cimport size_t
from cpython.ref cimport PyObject
cimport numpy as np
import numpy as np

cdef extern from "hermitian_grid.h":
    ctypedef struct complex32:
        float real
        float imag

    # import the C function with its new name
    void c_generate_hermitian_grid(
        size_t Nx, size_t Ny, size_t Nz,
        float boxlen,
        complex32* delta_k_3d,
        PyObject* py_power_spectrum
    )

def generate_hermitian_grid(int Nx, int Ny, int Nz, float boxlen, power_spectrum_func):
    """
    Python-facing wrapper. Returns a NumPy complex64[ Nx,Ny,Nz ] array.
    Accepts any Python callable power_spectrum_func(k) -> float.
    """
    # allocate numpy array
    cdef np.ndarray[np.complex64_t, ndim=3] delta_k = np.zeros((Nx, Ny, Nz), dtype=np.complex64)
    cdef complex32* ptr = <complex32*> delta_k.data

    # call the C function
    c_generate_hermitian_grid(Nx, Ny, Nz, boxlen, ptr, <PyObject*>power_spectrum_func)

    return delta_k
