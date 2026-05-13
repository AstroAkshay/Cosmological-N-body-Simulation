#include "update_x_core.h"
#include <stdint.h>
#include <math.h>
#include <omp.h>

void update_x_core(
    int64_t MM,
    int N1, int N2, int N3,
    double a,
    double delta_a,
    double coeff_x,
    double *rra,
    double *vva
) {
    int64_t p;
#pragma omp parallel for schedule(static)
    for(p = 0; p < MM; p++) {
        rra[3*p + 0] += coeff_x * vva[3*p + 0];
        rra[3*p + 1] += coeff_x * vva[3*p + 1];
        rra[3*p + 2] += coeff_x * vva[3*p + 2];

        // wrap positions into box
        rra[3*p + 0] = fmod(rra[3*p + 0] + N1, N1);
        rra[3*p + 1] = fmod(rra[3*p + 1] + N2, N2);
        rra[3*p + 2] = fmod(rra[3*p + 2] + N3, N3);
    }
}