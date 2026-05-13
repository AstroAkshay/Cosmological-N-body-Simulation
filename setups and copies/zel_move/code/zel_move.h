#ifndef ZEL_MOVE_H
#define ZEL_MOVE_H

#ifdef __cplusplus
extern "C" {
#endif

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
    int NF,
    double LL
);

#ifdef __cplusplus
}
#endif

#endif