#ifndef CALPOW_H
#define CALPOW_H

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _MSC_VER
    #ifndef _Complex
        #define _Complex
    #endif
    #pragma warning(push)
    #pragma warning(disable : 4201 4204 4996)
#endif

#include <fftw3.h>
#include <math.h>
#include <stddef.h>

// ======================================================================
//  Function: calpow_from_k
//  Purpose : Compute isotropic power spectrum P(k) from complex FFT field δ(k)
//  Input   :
//      N1, N2, N3  -> grid dimensions
//      Nbin        -> number of logarithmic bins
//      delta_k     -> pointer to FFTW complex array (r2c format)
//      tpibyL      -> 2π / L (wavenumber scaling)
//      vol         -> physical volume of the box (L³)
//  Output  :
//      power[Nbin] -> averaged power spectrum per bin
//      kmode[Nbin] -> mean |k| per bin
//      no[Nbin]    -> number of weighted modes per bin
// ======================================================================
void calpow_from_k(
    int N1, int N2, int N3,
    int Nbin,
    double* power,
    double* kmode,
    long* no,
    fftw_complex* delta_k,
    double tpibyL,
    double vol
);

#ifdef _MSC_VER
    #pragma warning(pop)
#endif

#ifdef __cplusplus
}
#endif

#endif // CALPOW_H
