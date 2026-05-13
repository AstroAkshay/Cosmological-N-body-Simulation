/*
 * Apply Zel'dovich displacement to a uniform lattice of particles
 *
 * Inputs:
 *   vfac  : growth factor for velocity
 *   va_x, va_y, va_z : displacement fields (flattened in row-major order)
 *   N1,N2,N3 : grid size along each axis
 *   NF    : subsampling factor (particle lattice spacing)
 *
 * Outputs:
 *   rra   : displaced particle positions (flattened, 3 per particle)
 *   vva   : velocities (flattened, 3 per particle)
 */
#include <math.h>
#include <omp.h>

void Zel_move_gradphi_vfac(
    double vfac,
    double *rra,
    double *vva,
    double *va_x,
    double *va_y,
    double *va_z,
    int N1,
    int N2,
    int N3,
    int NF
) {
    long jj, kk, ll, pin, idx;

#pragma omp parallel for collapse(3) private(jj,kk,ll,pin,idx)
    for(jj = 0; jj < N1 / NF; jj++)
        for(kk = 0; kk < N2 / NF; kk++)
            for(ll = 0; ll < N3 / NF; ll++) {
                pin = jj * (N2 / NF) * (N3 / NF) + kk * (N3 / NF) + ll;
                idx = (NF * jj) * N2 * N3 + (NF * kk) * N3 + (NF * ll);

                rra[3*pin + 0] = NF*jj + va_x[idx];
                rra[3*pin + 1] = NF*kk + va_y[idx];
                rra[3*pin + 2] = NF*ll + va_z[idx];

                vva[3*pin + 0] = vfac * va_x[idx];
                vva[3*pin + 1] = vfac * va_y[idx];
                vva[3*pin + 2] = vfac * va_z[idx];

                // Periodic BC with safety for negatives
                rra[3*pin + 0] = fmod(rra[3*pin + 0] + N1, N1);
                if (rra[3*pin + 0] < 0) rra[3*pin + 0] += N1;

                rra[3*pin + 1] = fmod(rra[3*pin + 1] + N2, N2);
                if (rra[3*pin + 1] < 0) rra[3*pin + 1] += N2;

                rra[3*pin + 2] = fmod(rra[3*pin + 2] + N3, N3);
                if (rra[3*pin + 2] < 0) rra[3*pin + 2] += N3;
            }
}