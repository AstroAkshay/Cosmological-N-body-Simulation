#include <math.h>
#include <omp.h>
#include <stddef.h>
#include "cic.h"

void cic(double *rra, double *ro,
         size_t N1, size_t N2, size_t N3,
         size_t MM, double rho_b_inv)
{
    const size_t NG = N1 * N2 * N3;

    ptrdiff_t i;

    /* Clear density grid */
    #pragma omp parallel for
    for (i = 0; i < (ptrdiff_t)NG; i++)
    {
        ro[i] = 0.0;
    }

    ptrdiff_t pin;

    /* Deposit particles */
    #pragma omp parallel for
    for (pin = 0; pin < (ptrdiff_t)MM; pin++)
    {
        double x = rra[3*pin + 0];
        double y = rra[3*pin + 1];
        double z = rra[3*pin + 2];

        ptrdiff_t ii = (ptrdiff_t)floor(x);
        ptrdiff_t jj = (ptrdiff_t)floor(y);
        ptrdiff_t kk = (ptrdiff_t)floor(z);

        double dx = x - (double)ii;
        double dy = y - (double)jj;
        double dz = z - (double)kk;

        ptrdiff_t ii1 = ii + 1;
        ptrdiff_t jj1 = jj + 1;
        ptrdiff_t kk1 = kk + 1;

        size_t ix  = (size_t)((ii  + (ptrdiff_t)N1) % N1);
        size_t jy  = (size_t)((jj  + (ptrdiff_t)N2) % N2);
        size_t kz  = (size_t)((kk  + (ptrdiff_t)N3) % N3);

        size_t ix1 = (size_t)((ii1 + (ptrdiff_t)N1) % N1);
        size_t jy1 = (size_t)((jj1 + (ptrdiff_t)N2) % N2);
        size_t kz1 = (size_t)((kk1 + (ptrdiff_t)N3) % N3);

        double w000 = (1-dx)*(1-dy)*(1-dz);
        double w100 = dx*(1-dy)*(1-dz);
        double w010 = (1-dx)*dy*(1-dz);
        double w110 = dx*dy*(1-dz);
        double w001 = (1-dx)*(1-dy)*dz;
        double w101 = dx*(1-dy)*dz;
        double w011 = (1-dx)*dy*dz;
        double w111 = dx*dy*dz;

        size_t idx000 = (ix  * N2 + jy ) * N3 + kz;
        size_t idx100 = (ix1 * N2 + jy ) * N3 + kz;
        size_t idx010 = (ix  * N2 + jy1) * N3 + kz;
        size_t idx110 = (ix1 * N2 + jy1) * N3 + kz;
        size_t idx001 = (ix  * N2 + jy ) * N3 + kz1;
        size_t idx101 = (ix1 * N2 + jy ) * N3 + kz1;
        size_t idx011 = (ix  * N2 + jy1) * N3 + kz1;
        size_t idx111 = (ix1 * N2 + jy1) * N3 + kz1;

        #pragma omp atomic
        ro[idx000] += w000 * rho_b_inv;

        #pragma omp atomic
        ro[idx100] += w100 * rho_b_inv;

        #pragma omp atomic
        ro[idx010] += w010 * rho_b_inv;

        #pragma omp atomic
        ro[idx110] += w110 * rho_b_inv;

        #pragma omp atomic
        ro[idx001] += w001 * rho_b_inv;

        #pragma omp atomic
        ro[idx101] += w101 * rho_b_inv;

        #pragma omp atomic
        ro[idx011] += w011 * rho_b_inv;

        #pragma omp atomic
        ro[idx111] += w111 * rho_b_inv;
    }
}