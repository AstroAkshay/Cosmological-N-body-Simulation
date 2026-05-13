#include "poisson_kernel.h"
#include <math.h>
#include <omp.h>

#ifndef PI
#define PI 3.14159265358979323846
#endif

void get_phi_k(
    double *delta_k,
    int N1,
    int N2,
    int N3,
    double Lbox,
    double vol
){
    int ii, jj, kk;
    int index, base;
    double kx, ky, kz, k2;

    #pragma omp parallel for private(jj,kk,index,base,kx,ky,kz,k2) schedule(static)
    for (ii = 0; ii < N1; ii++) {

        int ni = (ii <= N1/2) ? ii : ii - N1;
        kx = 2.0 * PI * ni / Lbox;
        double kx2 = kx * kx;

        for (jj = 0; jj < N2; jj++) {

            int nj = (jj <= N2/2) ? jj : jj - N2;
            ky = 2.0 * PI * nj / Lbox;
            double ky2 = ky * ky;

            for (kk = 0; kk <= N3/2; kk++) {

                kz = 2.0 * PI * kk / Lbox;
                double kz2 = kz * kz;

                index = (ii * N2 + jj) * (N3/2 + 1) + kk;
                base = 2 * index;

                k2 = kx2 + ky2 + kz2;

                if (k2 > 1e-14) {
                    delta_k[base]     *= -1.0 / k2;
                    delta_k[base + 1] *= -1.0 / k2;
                } else {
                    delta_k[base]     = 0.0;
                    delta_k[base + 1] = 0.0;
                }
            }
        }
    }
}