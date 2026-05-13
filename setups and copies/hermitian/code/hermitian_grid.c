#include "hermitian_grid.h"
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <Python.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static float gaussian_rand() {
    static int has_cached = 0;
    static float cached_value;
    if (has_cached) {
        has_cached = 0;
        return cached_value;
    }

    float u1, u2;
    do {
        u1 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 1.0f);
        u2 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 1.0f);
    } while (u1 <= 0.0f);

    float mag = sqrtf(-2.0f * logf(u1));
    float z0 = mag * cosf(2.0f * M_PI * u2);
    float z1 = mag * sinf(2.0f * M_PI * u2);

    cached_value = z1;
    has_cached = 1;
    return z0;
}

static inline complex32 conj_complex32(complex32 z) {
    complex32 c;
    c.real = z.real;
    c.imag = -z.imag;
    return c;
}

static inline size_t idx3d(size_t x, size_t y, size_t z, size_t Ny, size_t Nz) {
    return x * Ny * Nz + y * Nz + z;
}

/* Generate Hermitian cube (full array) */
void c_generate_hermitian_grid(size_t Nx, size_t Ny, size_t Nz,
                               float boxlen,
                               complex32* delta_k_3d,
                               PyObject* py_power_spectrum)
{
    srand((unsigned int)time(NULL));

    size_t i, j, k, ii, jj, kk, index1, index2;
    float amplitude, kx, ky, kz, k_val, Pk_val;
    int m;

    /* Fill all points, lines, planes, cubes */
    for (i = 0; i <= Nx / 2; i++) {
        for (j = 0; j <= Ny / 2; j++) {
            for (k = 0; k <= Nz / 2; k++) {

                /* Compute wavenumber */
                kx = (i <= Nx / 2 ? (float)i : (float)(i - Nx)) * (2.0f * M_PI / boxlen);
                ky = (j <= Ny / 2 ? (float)j : (float)(j - Ny)) * (2.0f * M_PI / boxlen);
                kz = (k <= Nz / 2 ? (float)k : (float)(k - Nz)) * (2.0f * M_PI / boxlen);
                k_val = sqrtf(kx*kx + ky*ky + kz*kz);

                /* P(k) */
                Pk_val = 1.0f;
                if (py_power_spectrum && PyCallable_Check(py_power_spectrum)) {
                    PyGILState_STATE gstate = PyGILState_Ensure();
                    PyObject* arg = Py_BuildValue("(f)", k_val);
                    PyObject* result = PyObject_CallObject(py_power_spectrum, arg);
                    Py_DECREF(arg);
                    if (result) {
                        Pk_val = (float)PyFloat_AsDouble(result);
                        Py_DECREF(result);
                    } else {
                        PyErr_Print();
                        PyErr_Clear();
                    }
                    PyGILState_Release(gstate);
                }

                amplitude = sqrtf(fmaxf(Pk_val, 0.0f) / 2.0f);

                /* Fill the "corner points" */
                index1 = idx3d(i, j, k, Ny, Nz);
                delta_k_3d[index1].real = amplitude * gaussian_rand();
                delta_k_3d[index1].imag = amplitude * gaussian_rand();

                /* Mirror across all axes to enforce Hermitian symmetry */
                for (m = 0; m < 8; m++) {
                    ii = (m & 1) ? Nx - i : i;
                    jj = (m & 2) ? Ny - j : j;
                    kk = (m & 4) ? Nz - k : k;

                    index2 = idx3d(ii % Nx, jj % Ny, kk % Nz, Ny, Nz);

                    if (index2 != index1) {
                        delta_k_3d[index2].real = delta_k_3d[index1].real;
                        delta_k_3d[index2].imag = -delta_k_3d[index1].imag;
                    }
                }
            }
        }
    }
}
