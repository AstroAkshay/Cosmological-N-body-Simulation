#include <math.h>
#include <omp.h>
#include "grad_phi.h"

/*
 * Compute v_i(k) = i * k_i / k^2 * delta(k)
 *
 * ro : input complex array (delta_k), FFTW r2c layout
 * va : output complex array (same layout)
 *
 * Array is interpreted as interleaved doubles:
 *   real = arr[2*index]
 *   imag = arr[2*index + 1]
 */

void grad_phi(int ix,
              double* ro, double* va,
              int N1, int N2, int N3,
              double Cx, double Cy, double Cz,
              double vol)
{
    long ii, jj, kk;
    long index;
    double AA, a0, a1, a2;
    double kix;

#pragma omp parallel for private(jj,kk,a0,a1,a2,AA,kix,index)
    for(ii = 0; ii < N1; ii++) {

        a0 = (ii > N1/2) ? Cx * (ii - N1) : Cx * ii;

        for(jj = 0; jj < N2; jj++) {

            a1 = (jj > N2/2) ? Cy * (jj - N2) : Cy * jj;

            for(kk = 0; kk < N3/2 + 1; kk++) {

                a2 = kk * Cz;
                index = (ii * N2 + jj) * (N3/2 + 1) + kk;

                AA = a0*a0 + a1*a1 + a2*a2;

                if (AA < 1e-12) {
                    va[2*index]     = 0.0;
                    va[2*index + 1] = 0.0;
                    continue;
                }

                switch(ix) {
                    case 0: kix = a0; break;
                    case 1: kix = a1; break;
                    case 2: kix = a2; break;
                    default: kix = 0.0;
                }

                AA = kix / AA;

                /* Multiply by i and divide by volume */
                va[2*index]     = -AA * ro[2*index + 1] / vol;
                va[2*index + 1] =  AA * ro[2*index]     / vol;
            }
        }
    }
}
