#ifndef FFTGRID_H
#define FFTGRID_H

#include <stddef.h>

typedef struct {
    float real;
    float imag;
} complex32;

void generate_delta_k_3d(size_t Nx, size_t Ny, size_t Nz, complex32* delta_k_3d);

#endif // FFTGRID_H
