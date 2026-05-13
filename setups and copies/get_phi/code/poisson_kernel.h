#ifndef POISSON_KERNEL_H
#define POISSON_KERNEL_H

#ifdef __cplusplus
extern "C" {
#endif

void get_phi_k(
    double *delta_k,
    int N1,
    int N2,
    int N3,
    double Lbox,
    double vol
);

#ifdef __cplusplus
}
#endif

#endif
