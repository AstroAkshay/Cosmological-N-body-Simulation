#ifndef GLOBALS_H
#define GLOBALS_H

#include <fftw3.h>

/* Grid dimensions */
extern int N1, N2, N3;

/* k-space constants */
extern float Cx, Cy, Cz;
extern float vol;

/* FFTW arrays */
extern fftwf_complex ***ro;   /* delta_k */
extern fftwf_complex ***va;   /* output */

#endif
