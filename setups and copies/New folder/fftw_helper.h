#ifndef FFTW_HELPER_H
#define FFTW_HELPER_H

#ifdef _MSC_VER
#include <complex.h>
typedef _Dcomplex fftw_complex;
#else
#include <complex.h>
typedef double complex fftw_complex;
#endif

typedef void* fftw_plan;

// Manual DLL import declarations for MSVC
__declspec(dllimport) fftw_plan fftw_plan_dft_3d(
    int n0, int n1, int n2,
    fftw_complex *in, fftw_complex *out,
    int sign, unsigned flags);
__declspec(dllimport) void fftw_execute(fftw_plan p);
__declspec(dllimport) void fftw_destroy_plan(fftw_plan p);

__declspec(dllimport) fftw_plan fftw_plan_dft_r2c_3d(
    int n0, int n1, int n2,
    double *in, fftw_complex *out,
    unsigned flags);
__declspec(dllimport) fftw_plan fftw_plan_dft_c2r_3d(
    int n0, int n1, int n2,
    fftw_complex *in, double *out,
    unsigned flags);

// FFT direction constants
#define FFTW_FORWARD (-1)
#define FFTW_BACKWARD (+1)
#define FFTW_ESTIMATE (1U << 6)

#endif // FFTW_HELPER_H
