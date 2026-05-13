# cython: boundscheck=False, wraparound=False, cdivision=True

import numpy as np
cimport numpy as np

cdef extern from "grad_phi.h":
    void grad_phi(int ix,
                  double* ro, double* va,
                  int N1, int N2, int N3,
                  double Cx, double Cy, double Cz,
                  double vol)

def grad_phi_py(
    int ix,
    np.ndarray[np.complex128_t, ndim=3, mode="c"] ro,
    np.ndarray[np.complex128_t, ndim=3, mode="c"] va,
    int N1, int N2, int N3,
    double Cx, double Cy, double Cz,
    double vol
):

    if ro.shape[0] != N1 or ro.shape[1] != N2 or ro.shape[2] != N3//2 + 1:
        raise ValueError("ro has incorrect shape")

    if va.shape[0] != N1 or va.shape[1] != N2 or va.shape[2] != N3//2 + 1:
        raise ValueError("va has incorrect shape")

    grad_phi(
        ix,
        <double*> ro.data,
        <double*> va.data,
        N1, N2, N3,
        Cx, Cy, Cz,
        vol
    )

    return va