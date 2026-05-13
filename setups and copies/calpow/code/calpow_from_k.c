#include "calpow.h"
#include <stdio.h>
#include <math.h>

void calpow_from_k(
    int N1, int N2, int N3,
    int Nbin,
    double* power,
    double* kmode,
    long* no,
    fftw_complex* delta_k,   // <-- switched to double-precision FFTW complex
    double tpibyL,
    double vol
) {
    long i, j, k, index;
    double fac1 = 1.0 / (N1 * N1);
    double fac2 = 1.0 / (N2 * N2);
    double fac3 = 1.0 / (N3 * N3);
    double scale = log10(0.5 * N1 + 1e-8) / Nbin;

    // Reset bins
    for (i = 0; i < Nbin; i++) {
        power[i] = 0.0;
        kmode[i] = 0.0;
        no[i] = 0;
    }

    // Loop through half-cube (r2c layout)
    for (i = 0; i < N1; i++) {
        int a = (i > N1 / 2) ? N1 - i : i;
        for (j = 0; j < N2; j++) {
            int b = (j > N2 / 2) ? N2 - j : j;
            for (k = 0; k <= N3 / 2; k++) {
                int c = k;
                index = i * N2 * (N3 / 2 + 1) + j * (N3 / 2 + 1) + k;

                double m = sqrt(fac1 * a * a + fac2 * b * b + fac3 * c * c);
                if (m == 0.0) continue;

                // --- Symmetry correction weight ---
                double weight = 1.0;
                if ((i == 0 || i == N1 / 2) && (j == 0 || j == N2 / 2) && (k == 0 || k == N3 / 2))
                    weight = 1.0;   // corner/line mode
                else if (((i == 0 || i == N1 / 2) && (j == 0 || j == N2 / 2)) ||
                         ((i == 0 || i == N1 / 2) && (k == 0 || k == N3 / 2)) ||
                         ((j == 0 || j == N2 / 2) && (k == 0 || k == N3 / 2)))
                    weight = 2.0;   // half-plane
                else if ((i == 0 || i == N1 / 2) ||
                         (j == 0 || j == N2 / 2) ||
                         (k == 0 || k == N3 / 2))
                    weight = 4.0;   // half-line
                else
                    weight = 8.0;   // interior

                // --- Compute bin index ---
                int d = (int)floor(log10(m * N1 + 1e-12) / scale);
                if (d < 0 || d >= Nbin) continue;

                // --- Power computation (double precision) ---
                double re = delta_k[index][0];
                double im = delta_k[index][1];
                double pk = weight * (re * re + im * im);

                power[d] += pk;
                kmode[d] += m * weight;
                no[d] += (long)weight;
            }
        }
    }

    // Normalize results
    for (i = 0; i < Nbin; i++) {
        if (no[i] > 0) {
            power[i] /= (no[i] * vol);
            kmode[i] = tpibyL * kmode[i] / no[i];
        }
    }
}
