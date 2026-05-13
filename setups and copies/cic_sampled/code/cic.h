#ifndef CIC_H
#define CIC_H

#include <stddef.h>   /* ptrdiff_t */

void cic_sampled_flat(
    double *rra,
    int *s_indx,
    ptrdiff_t MM,
    double *ro,
    ptrdiff_t N1,
    ptrdiff_t N2,
    ptrdiff_t N3,
    double rho_b_inv
);

#endif
