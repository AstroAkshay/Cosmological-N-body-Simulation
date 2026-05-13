#include <stdio.h>
#include <math.h>
#include <fftw3.h>

int main() {
    int N = 8;
    double in[N];
    fftw_complex out[N/2 + 1];
    fftw_plan plan;

    // initialize sample data
    for (int i = 0; i < N; i++)
        in[i] = sin(2 * M_PI * i / N);

    // create FFT plan
    plan = fftw_plan_dft_r2c_1d(N, in, out, FFTW_ESTIMATE);

    // execute FFT
    fftw_execute(plan);

    printf("FFT output:\n");
    for (int i = 0; i < N/2 + 1; i++)
        printf("%2d: (%f, %f)\n", i, out[i][0], out[i][1]);

    fftw_destroy_plan(plan);
    fftw_cleanup();
    return 0;
}
