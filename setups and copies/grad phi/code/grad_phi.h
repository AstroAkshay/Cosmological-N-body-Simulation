#ifndef GRAD_PHI_H
#define GRAD_PHI_H

void grad_phi(int ix,
              double* ro, double* va,
              int N1, int N2, int N3,
              double Cx, double Cy, double Cz,
              double vol);

#endif
