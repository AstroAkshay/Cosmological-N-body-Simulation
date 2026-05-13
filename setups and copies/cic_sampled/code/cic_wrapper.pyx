# distutils: language = c
# distutils: extra_compile_args = /openmp

import numpy as np
cimport numpy as np

cdef extern from "cic.h":
    void cic_sampled_flat(
        double *rra,
        int *s_indx,
        Py_ssize_t MM,
        double *ro,
        Py_ssize_t N1,
        Py_ssize_t N2,
        Py_ssize_t N3,
        double rho_b_inv
    )

def cic_sampled_py(
    np.ndarray[np.float64_t, ndim=2, mode="c"] rra,
    np.ndarray[np.int32_t,   ndim=1, mode="c"] s_indx,
    np.ndarray[np.float64_t, ndim=3, mode="c"] ro,
    double rho_b_inv
):
    """
    rra     : (MM, 3) particle positions
    s_indx  : (MM,) int32 (-1 = include)
    ro      : (N1, N2, N3) output density grid
    """

    cdef Py_ssize_t MM = rra.shape[0]
    cdef Py_ssize_t N1 = ro.shape[0]
    cdef Py_ssize_t N2 = ro.shape[1]
    cdef Py_ssize_t N3 = ro.shape[2]

    cic_sampled_flat(
        <double*> rra.data,
        <int*>    s_indx.data,
        MM,
        <double*> ro.data,
        N1, N2, N3,
        rho_b_inv
    )
