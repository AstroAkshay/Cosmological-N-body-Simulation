#ifndef CIC_H
#define CIC_H

#include <stddef.h>

void cic(double *rra, double *ro,
         size_t N1, size_t N2, size_t N3,
         size_t MM, double rho_b_inv);

#endif
