#include "update_v_core.h"
#include <math.h>
#include <stdint.h>
#include <omp.h>

#define IDX(i,j,k,N2,N3) (((i)*(N2)+(j))*(N3)+(k))

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
) {
    double coeff, vflag;
    double dx = LL / N1;

    if(delta_a > 0.0) {
        // Proper cosmological scaling: a^2 in denominator
        coeff = 1.5 * Omega_m * delta_a / (a * a * Hf);
        vflag = 1.0;
    } else {
        // Zel'dovich initialization
        coeff = a * a * Hf;
        vflag = 0.0;
    }

    coeff /= (2.0 * dx);  // central difference scaling

    int64_t p;

#pragma omp parallel for schedule(static)
    for(p = 0; p < MM; p++) {

        double x = rra[3*p + 0];
        double y = rra[3*p + 1];
        double z = rra[3*p + 2];

        int a_idx = (int)floor(x);
        int b_idx = (int)floor(y);
        int c_idx = (int)floor(z);

        double g0 = 0.0, g1 = 0.0, g2 = 0.0;

        int ii, jj, kk;
        int ix, jy, kz;
        int xp, xn, yp, yn, zp, zn;
        double wx, wy, wz;

        for(ii=0; ii<=1; ii++) {
            ix = (a_idx + ii) % N1;
            xp = (ix - 1 + N1) % N1;
            xn = (ix + 1) % N1;
            wx = fabs(1.0 - x + a_idx - ii);

            for(jj=0; jj<=1; jj++) {
                jy = (b_idx + jj) % N2;
                yp = (jy - 1 + N2) % N2;
                yn = (jy + 1) % N2;
                wy = fabs(1.0 - y + b_idx - jj);

                for(kk=0; kk<=1; kk++) {
                    kz = (c_idx + kk) % N3;
                    zp = (kz - 1 + N3) % N3;
                    zn = (kz + 1) % N3;
                    wz = fabs(1.0 - z + c_idx - kk);

                    g0 += wx * wy * wz * (phi[IDX(xp,jy,kz,N2,N3)] - phi[IDX(xn,jy,kz,N2,N3)]);
                    g1 += wx * wy * wz * (phi[IDX(ix,yp,kz,N2,N3)] - phi[IDX(ix,yn,kz,N2,N3)]);
                    g2 += wx * wy * wz * (phi[IDX(ix,jy,zp,N2,N3)] - phi[IDX(ix,jy,zn,N2,N3)]);
                }
            }
        }

        vva[3*p + 0] = vflag*vva[3*p + 0] + coeff*g0;
        vva[3*p + 1] = vflag*vva[3*p + 1] + coeff*g1;
        vva[3*p + 2] = vflag*vva[3*p + 2] + coeff*g2;
    }
}