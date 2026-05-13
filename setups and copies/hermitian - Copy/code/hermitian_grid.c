#include "hermitian_grid.h"
#include <stdlib.h>
#include <math.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Gaussian random number generator (Box-Muller)
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

// 3D to 1D index mapping
static inline size_t idx3d(size_t x, size_t y, size_t z, size_t Ny, size_t Nz) {
    return x * Ny * Nz + y * Nz + z;
}

// Generate Hermitian-symmetric 3D Fourier grid
void generate_hermitian_grid(size_t Nx, size_t Ny, size_t Nz, complex32* delta_k_3d) {
    srand((unsigned int)time(NULL));

    size_t x, y, z;
    for (x = 0; x < Nx; x++) {
        size_t xm = (Nx - x) % Nx;
        for (y = 0; y < Ny; y++) {
            size_t ym = (Ny - y) % Ny;
            for (z = 0; z < Nz; z++) {
                size_t zm = (Nz - z) % Nz;
                size_t i = idx3d(x, y, z, Ny, Nz);

                // Self-conjugate points -> purely real
                if ((x == 0 || x == Nx / 2) &&
                    (y == 0 || y == Ny / 2) &&
                    (z == 0 || z == Nz / 2)) {
                    delta_k_3d[i].real = gaussian_rand();
                    delta_k_3d[i].imag = 0.0f;
                }
                // Independent half (only fill once)
                else if (x < Nx / 2 ||
                         (x == Nx / 2 && y < Ny / 2) ||
                         (x == Nx / 2 && y == Ny / 2 && z < Nz / 2)) {
                    delta_k_3d[i].real = gaussian_rand();
                    delta_k_3d[i].imag = gaussian_rand();

                    // Fill conjugate pair
                    size_t j = idx3d(xm, ym, zm, Ny, Nz);
                    delta_k_3d[j] = conj_complex32(delta_k_3d[i]);
                }
                // else: conjugate point already filled
            }
        }
    }
}
