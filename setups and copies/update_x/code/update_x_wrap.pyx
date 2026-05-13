# cython: boundscheck=False, wraparound=False
# distutils: extra_compile_args = /openmp

import numpy as np
cimport numpy as np
from libc.stdint cimport int64_t

cdef extern from "update_x_core.h":
    void update_x_core(
        int64_t MM,
        int N1, int N2, int N3,
        double a,
        double delta_a,
        double coeff_x,
        double *rra,
        double *vva
    )

def py_update_x(
    int64_t MM,                  # number of particles
    int N1, int N2, int N3,      # grid sizes
    double a,                    # current scale factor
    double delta_a,              # step
    double coeff_x,              # drift coefficient
    np.ndarray[np.float64_t, ndim=1] rra,  # flattened positions (3*MM,)
    np.ndarray[np.float64_t, ndim=1] vva   # flattened velocities (3*MM,)
):
    """
    Python wrapper for update_x_core using 1D flattened arrays
    """
    # ensure C-contiguous
    if not rra.flags['C_CONTIGUOUS']:
        rra = np.ascontiguousarray(rra, dtype=np.float64)
    if not vva.flags['C_CONTIGUOUS']:
        vva = np.ascontiguousarray(vva, dtype=np.float64)

    update_x_core(
        MM,
        N1, N2, N3,
        a,
        delta_a,
        coeff_x,
        <double*> rra.data,
        <double*> vva.data
    )