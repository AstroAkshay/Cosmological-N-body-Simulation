#include <math.h>
#include <omp.h>
#include <stddef.h>
#include "cic.h"

void cic_sampled_flat(
    double *rra,
    int *s_indx,
    ptrdiff_t MM,
    double *ro,
    ptrdiff_t N1,
    ptrdiff_t N2,
    ptrdiff_t N3,
    double rho_b_inv
)
{
    ptrdiff_t i, j, k, ix, jy, kz, pin, index;
    int ii, jj, kk;
    double wx, wy, wz;

    /* Clear density grid */
    #pragma omp parallel for
    for (i = 0; i < N1; i++)
        for (j = 0; j < N2; j++)
            for (k = 0; k < N3; k++) {
                index = (i * N2 + j) * N3 + k;
                ro[index] = 0.0;
            }

    /* Particle deposition */
    #pragma omp parallel for private(i,j,k,ii,jj,kk,wx,wy,wz,ix,jy,kz,index)
    for (pin = 0; pin < MM; pin++) {

        if (s_indx[pin] == -1) {

            double x = rra[3 * pin + 0];
            double y = rra[3 * pin + 1];
            double z = rra[3 * pin + 2];

            i = (ptrdiff_t)floor(x);
            j = (ptrdiff_t)floor(y);
            k = (ptrdiff_t)floor(z);

            for (ii = 0; ii <= 1; ii++) {
                wx = fabs(1.0 - x + i - ii) * rho_b_inv;
                ix = (i + ii + N1) % N1;

                for (jj = 0; jj <= 1; jj++) {
                    wy = fabs(1.0 - y + j - jj);
                    jy = (j + jj + N2) % N2;

                    for (kk = 0; kk <= 1; kk++) {
                        wz = fabs(1.0 - z + k - kk);
                        kz = (k + kk + N3) % N3;

                        index = (ix * N2 + jy) * N3 + kz;

                        #pragma omp atomic
                        ro[index] += wx * wy * wz;
                    }
                }
            }
        }
    }
}
