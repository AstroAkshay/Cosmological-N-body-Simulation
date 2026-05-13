# distutils: language = c
# cython: language_level = 3

import numpy as np
cimport numpy as np

# ---- Define fftw_complex manually for Cython ----
ctypedef double fftw_complex[2]

# ---- Declare external C function from calpow.c ----
cdef extern from "calpow.h":
    void calpow_from_k(
        int N1, int N2, int N3,
        int Nbin,
        double* power,
        double* kmode,
        long* no,
        fftw_complex* delta_k,
        double tpibyL,
        double vol
    )

def calpow(np.ndarray[np.complex128_t, ndim=3, mode="c"] delta_k,
           int Nbin,
           double tpibyL,
           double vol):
    """
    Compute isotropic power spectrum P(k) from Fourier-space field δ(k)
    using double precision (complex128).

    Parameters
    ----------
    delta_k : np.ndarray[np.complex128_t, ndim=3]
        Fourier-space complex array from FFTW (shape = [N1, N2, N3//2 + 1]).
        Must be real-to-complex half-spectrum layout (r2c).
    Nbin : int
        Number of logarithmic bins for the power spectrum.
    tpibyL : float
        2 * π / BoxLength (k-space scaling factor).
    vol : float
        Physical volume of the simulation box (L³).

    Returns
    -------
    power : np.ndarray[np.float64_t]
        Binned power spectrum P(k)
    kmode : np.ndarray[np.float64_t]
        Average |k| value in each bin
    no : np.ndarray[np.int64_t]
        Weighted number of modes per bin
    """

    # ---- Validate input ----
    if delta_k.ndim != 3:
        raise ValueError("delta_k must be a 3D complex array of shape [N1, N2, N3/2 + 1].")

    if not delta_k.flags["C_CONTIGUOUS"]:
        delta_k = np.ascontiguousarray(delta_k, dtype=np.complex128)

    # ---- Extract grid dimensions ----
    cdef int N1 = delta_k.shape[0]
    cdef int N2 = delta_k.shape[1]
    cdef int N3 = (delta_k.shape[2] - 1) * 2  # reconstruct full z-dimension

    # ---- Allocate outputs ----
    cdef np.ndarray[np.float64_t, ndim=1] power = np.zeros(Nbin, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] kmode = np.zeros(Nbin, dtype=np.float64)
    cdef np.ndarray[np.int64_t, ndim=1] no = np.zeros(Nbin, dtype=np.int64)

    # ---- Call high-accuracy C implementation ----
    calpow_from_k(
        N1, N2, N3, Nbin,
        <double*>power.data,
        <double*>kmode.data,
        <long*>no.data,
        <fftw_complex*>delta_k.data,
        tpibyL,
        vol
    )

    return power, kmode, no
