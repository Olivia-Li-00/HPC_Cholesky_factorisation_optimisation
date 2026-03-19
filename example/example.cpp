/**
 * @file:
 *      example.cpp
 * @brief:
 *      Example program demonstrating the usage of the Cholesky factorisation library
 *
 *      This program illustrates how a user would:
 *      1. Populate a symmetric positive-definite matrix (nxn) in a flattened 1D array
 *      2. Call the Cholesky factorisation in-place update routine to obtain the factorised matrix,
 *         in the lower-upper form: the lower-diagonal part of the output matrix contains elements of L,
 *         and the upper-diagonal part of the output matrix contains L^T.
 *         where C = L L^T
 *         and L is a lower-triangular matrix and L^T is its symmetric transpose,
 *      3. Use the resulting output matrix to compute the log-determinant of the original input matrix C via Equation 4 from the coursework
 *
 *      Note:
 *      The matrix size can be set via the command line (default: 5).
 */
#include "cholesky_factorisation.h"
#include "utils.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char *argv[]) {

    // Read matrix size from command line, default to 5
    long n = 5;
    if (argc > 1) {
        n = atol(argv[1]);
        if (n <= 0) {
            fprintf(stderr, "Error: Matrix size must be a positive integer.\n");
            return 1;
        }
    }

    printf("Cholesky factorisation library usage example\n");
    printf("Matrix size: %ldx%ld\n\n", n, n);

    // Allocate and populate a symmetric positive-definite matrix (1D flattened array of size nxn) which is used as input for the cholesky factorisation routine
    std::vector<double> c(n * n);
    populate_matrix(c.data(), n);

    // Optionally print the input matrix for small sizes
    if (n <= 5) {
        printf("Input matrix C:\n");
        for (long i = 0; i < n; i++) {
            for (long j = 0; j < n; j++) {
                printf("%10.4f ", c[n * i + j]);
            }
            printf("\n");
        }
        printf("\n");
    }

    // Perform Cholesky factorisation
    // On return, the output matrix stores L in the lower triangle and L^T in the upper triangle
    double time = cholesky_factorisation(c.data(), n);

    if (time < 0.0) {
        fprintf(stderr, "Cholesky factorisation failed.\n");
        return 1;
    }

    printf("Factorisation completed in %.6f seconds.\n", time);

    // Optionally print the output matrix for small sizes
    if (n <= 5) {
        printf("\nOutput matrix (L at the lower triangle, L^T at the upper triangle):\n");
        for (long i = 0; i < n; i++) {
            for (long j = 0; j < n; j++) {
                printf("%10.4f ", c[n * i + j]);
            }
            printf("\n");
        }
        printf("\n");
    }

    // Compute the log-determinant of input matrix using the output matrix using Equation 4 from the coursework
    double log_det = compute_log_det(c.data(), n);
    printf("log|C| = %.10f\n", log_det);

    return 0;
}