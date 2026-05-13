#ifndef HERMITIAN_GRID_H
#define HERMITIAN_GRID_H

#include <stddef.h>
#include <Python.h>

typedef struct {
    float real;
    float imag;
} complex32;

/* C function — renamed to avoid clash with Python wrapper */
void c_generate_hermitian_grid(
    size_t Nx, size_t Ny, size_t Nz,
    float boxlen,
    complex32* delta_k_3d,
    PyObject* py_power_spectrum
);

#endif
