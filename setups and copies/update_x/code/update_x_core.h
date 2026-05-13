#ifndef UPDATE_X_CORE_H
#define UPDATE_X_CORE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void update_x_core(
    int64_t MM,       // number of particles
    int N1, int N2, int N3,
    double a,
    double delta_a,
    double coeff_x,
    double *rra,      // particle positions, shape (MM,3)
    double *vva       // particle velocities, shape (MM,3)
);

#ifdef __cplusplus
}
#endif

#endif /* UPDATE_X_CORE_H */
