#include <stdio.h>
#include <math.h>
#include <fftw3.h>

int main() {
    // Grid size
    int N1 = 8, N2 = 8, N3 = 8;
    int Nreal = N1 * N2 * N3;
    int Ncomplex = N1 * N2 * (N3/2 + 1);

    // Allocate arrays
    float *rho = fftwf_alloc_real(Nreal);          // real space
    fftwf_complex *delta_k = fftwf_alloc_complex(Ncomplex);  // Fourier space

    // Create FFTW plans
    fftwf_plan plan_forward  = fftwf_plan_dft_r2c_3d(N1, N2, N3, rho, delta_k, FFTW_ESTIMATE);
    fftwf_plan plan_backward = fftwf_plan_dft_c2r_3d(N1, N2, N3, delta_k, rho, FFTW_ESTIMATE);

    // -------------------------------------------------------
    // 1. Initialize real-space array with some test data
    for (int i = 0; i < Nreal; i++) {
        rho[i] = (i % 16 == 0) ? 1.0f : 0.0f;  // simple pattern
    }

    // 2. Forward FFT: rho(x) -> delta_k(k)
    fftwf_execute(plan_forward);

    printf("Some Fourier coefficients:\n");
    for (int i = 0; i < 5; i++) {
        printf("delta_k[%d] = %f + %fi\n", i,
               delta_k[i][0], delta_k[i][1]);
    }

    // 3. Inverse FFT: delta_k(k) -> rho(x)
    fftwf_execute(plan_backward);

    // Normalize (FFTW does not normalize the transform)
    for (int i = 0; i < Nreal; i++) {
        rho[i] /= (N1 * N2 * N3);
    }

    printf("\nRecovered real-space values:\n");
    for (int i = 0; i < 16; i++) {
        printf("rho[%d] = %f\n", i, rho[i]);
    }

    // -------------------------------------------------------
    // Cleanup
    fftwf_destroy_plan(plan_forward);
    fftwf_destroy_plan(plan_backward);
    fftwf_free(rho);
    fftwf_free(delta_k);

    return 0;
}
