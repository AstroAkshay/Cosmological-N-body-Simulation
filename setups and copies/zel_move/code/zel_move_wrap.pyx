# distutils: language = c
# distutils: extra_compile_args = /openmp
# distutils: extra_link_args = /openmp

import numpy as np
cimport numpy as np

# Declare the C function
cdef extern from "zel_move.h" nogil:
    void Zel_move_gradphi_vfac(
        double vfac,
        double *rra,
        double *vva,
        double *va_x,
        double *va_y,
        double *va_z,
        int N1,
        int N2,
        int N3,
        int NF,
        double LL
    )

def py_Zel_move_gradphi(
    double vfac,
    np.ndarray[np.float64_t, ndim=1, mode="c"] rra,
    np.ndarray[np.float64_t, ndim=1, mode="c"] vva,
    np.ndarray[np.float64_t, ndim=1, mode="c"] va_x,
    np.ndarray[np.float64_t, ndim=1, mode="c"] va_y,
    np.ndarray[np.float64_t, ndim=1, mode="c"] va_z,
    int N1,
    int N2,
    int N3,
    int NF,
    double LL
):
    """
    Python wrapper for Zel_move_gradphi_vfac.
    All arrays are 1D flattened np.float64 arrays.
    """
    # Ensure contiguous arrays
    assert rra.flags['C_CONTIGUOUS']
    assert vva.flags['C_CONTIGUOUS']
    assert va_x.flags['C_CONTIGUOUS']
    assert va_y.flags['C_CONTIGUOUS']
    assert va_z.flags['C_CONTIGUOUS']

    # Get raw pointers
    cdef double *prra = &rra[0]
    cdef double *pvva = &vva[0]
    cdef double *pva_x = &va_x[0]
    cdef double *pva_y = &va_y[0]
    cdef double *pva_z = &va_z[0]

    # Call C function without GIL (safe because it uses OpenMP only)
    with nogil:
        Zel_move_gradphi_vfac(
            vfac,
            prra,
            pvva,
            pva_x,
            pva_y,
            pva_z,
            N1, N2, N3,
            NF,
            LL
        )