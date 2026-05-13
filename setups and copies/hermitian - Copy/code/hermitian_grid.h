#ifndef HERMITIAN_GRID_H
#define HERMITIAN_GRID_H

#include <stddef.h>

typedef struct {
    float real;
    float imag;
} complex32;

// Generate Hermitian-symmetric 3D grid
// Self-conjugate elements (like k=0 and Nyquist) have imag = 0
void generate_hermitian_grid(size_t Nx, size_t Ny, size_t Nz, complex32* delta_k_3d);

#endif // HERMITIAN_GRID_H
