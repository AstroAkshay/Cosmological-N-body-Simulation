#include "fftw_helper.h"

void fft3d_r2c(double *in, fftw_complex *out, int Nx, int Ny, int Nz) {
    fftw_plan plan = fftw_plan_dft_r2c_3d(Nx, Ny, Nz, in, out, FFTW_ESTIMATE);
    fftw_execute(plan);
    fftw_destroy_plan(plan);
}

void fft3d_c2r(fftw_complex *in, double *out, int Nx, int Ny, int Nz_full) {
    int Nz_complex = Nz_full/2 + 1;
    fftw_plan plan = fftw_plan_dft_c2r_3d(Nx, Ny, Nz_full, in, out, FFTW_ESTIMATE);
    fftw_execute(plan);
    fftw_destroy_plan(plan);
}
