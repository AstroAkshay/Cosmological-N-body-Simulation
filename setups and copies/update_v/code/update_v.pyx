# cython: boundscheck=False, wraparound=False
import numpy as np
cimport numpy as np
from libc.stdint cimport int64_t

cdef extern from "update_v_core.h":
    void update_v_core(
        int64_t MM,
        int N1, int N2, int N3,
        double a,
        double delta_a,
        double Omega_m,
        double Hf,
        double LL,
        double *rra,
        double *vva,
        double *phi
    )

def py_update_v(
    int64_t MM,
    int N1, int N2, int N3,
    double a,
    double delta_a,
    double Omega_m,
    double Hf,
    double LL,
    np.ndarray[np.float64_t, ndim=1] rra,
    np.ndarray[np.float64_t, ndim=1] vva,
    np.ndarray[np.float64_t, ndim=1] phi
):
    """
    Python wrapper for update_v_core.

    Arguments must be flattened 1D arrays:
    - rra, vva: shape (3*MM,)
    - phi: shape (N1*N2*N3,)
    """
    if not rra.flags['C_CONTIGUOUS']:
        raise ValueError("rra must be C-contiguous")
    if not vva.flags['C_CONTIGUOUS']:
        raise ValueError("vva must be C-contiguous")
    if not phi.flags['C_CONTIGUOUS']:
        raise ValueError("phi must be C-contiguous")

    update_v_core(
        MM,
        N1, N2, N3,
        a,
        delta_a,
        Omega_m,
        Hf,
        LL,
        <double *> rra.data,
        <double *> vva.data,
        <double *> phi.data
    )