# distutils: language = c
# distutils: extra_compile_args = /openmp

import numpy as np
cimport numpy as np
from libc.stddef cimport size_t

cdef extern from "cic.h":
    void cic(double *rra, double *ro,
             size_t N1, size_t N2, size_t N3,
             size_t MM, double rho_b_inv)

def cic_py(np.ndarray[np.float64_t, ndim=2] rra,
           size_t N1, size_t N2, size_t N3,
           double rho_b_inv):

    cdef size_t MM = rra.shape[0]

    # allocate density grid
    ro_np = np.zeros((N1, N2, N3), dtype=np.float64)

    # typed memoryviews
    cdef double[:, :] rra_mv = rra
    cdef double[:, :, :] ro_mv = ro_np

    cic(&rra_mv[0,0],
        &ro_mv[0,0,0],
        N1, N2, N3,
        MM, rho_b_inv)

    return ro_np