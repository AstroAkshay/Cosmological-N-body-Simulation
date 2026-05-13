# distutils: language = c
# distutils: extra_compile_args = /openmp   # ← ONLY LINE ADDED

import numpy as np
cimport numpy as np

cdef extern from "poisson_kernel.h":
    void get_phi_k(
        double* delta_k,
        int N1,
        int N2,
        int N3,
        double Lbox,
        double vol
    )

def apply_inverse_laplacian(np.ndarray[np.complex128_t, ndim=3] delta_k,
                            double Lbox,
                            double vol):
    """
    In-place inverse Laplacian in Fourier space.
    FFT must be performed outside.
    """

    if not delta_k.flags["C_CONTIGUOUS"]:
        raise ValueError("delta_k must be C-contiguous")

    cdef Py_ssize_t s0 = delta_k.shape[0]
    cdef Py_ssize_t s1 = delta_k.shape[1]
    cdef Py_ssize_t s2 = delta_k.shape[2]

    if s0 > 2**31 - 1 or s1 > 2**31 - 1 or s2 > 2**31 - 1:
        raise ValueError("Grid too large for int-based kernel")

    cdef int N1 = <int> s0
    cdef int N2 = <int> s1
    cdef int N3 = <int> ((s2 - 1) * 2)

    get_phi_k(
        <double*> delta_k.data,
        N1,
        N2,
        N3,
        Lbox,
        vol
    )

    return delta_k
