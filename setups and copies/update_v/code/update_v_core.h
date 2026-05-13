#ifndef UPDATE_V_CORE_H
#define UPDATE_V_CORE_H

#include <stdint.h>

/*
  Update particle velocities using finite-difference CIC gradient.
  MM: number of particles
  N1,N2,N3: grid dimensions
  a: scale factor
  delta_a: timestep
  Omega_m: matter density
  Hf: Hubble factor
  LL: box size
  rra: particle positions [MM x 3]
  vva: particle velocities [MM x 3]
  phi: 3D potential/density grid [N1 x N2 x N3] computed from CIC
*/
void update_v_core(
    int64_t MM,
    int N1, int N2, int N3,
    double a,
    double delta_a,
    double Omega_m,
    double Hf,
    double LL,
    double *rra,
    double *vva,
    double *phi
);

#endif
