#include "fftgrid.h"
#include <stdlib.h>
#include <math.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Gaussian random number (Box-Muller)
static float gaussian_rand() {
    float u1 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 1.0f);
    float u2 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 1.0f);
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
}

// Complex conjugate
static inline complex32 conj_complex32(complex32 z) {
    complex32 c;
    c.real = z.real;
    c.imag = -z.imag;
    return c;
}

void generate_delta_k_3d(size_t Nx, size_t Ny, size_t Nz, complex32* delta_k_3d) {
    size_t total_points = Nx * Ny * Nz;
    size_t n_half = total_points / 2;

    srand((unsigned int)time(NULL));

    // Fill first half
    for (size_t i = 0; i < n_half; i++) {
        delta_k_3d[i].real = gaussian_rand();
        delta_k_3d[i].imag = gaussian_rand();
    }

    // Fill second half with complex conjugates
    for (size_t i = n_half; i < total_points; i++) {
        delta_k_3d[i] = conj_complex32(delta_k_3d[i - n_half]);
    }
}
