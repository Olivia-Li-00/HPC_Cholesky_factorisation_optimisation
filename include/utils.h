/**
 * @file:
        utils.h
 * @brief:
 *      Shared utility for correlation matrix generation and validation
 *
 *      Provides helper functions commonly used across the project for:
 *      1. Computing pairwise correlation values
 *      2. Populating symmetric positive-definite correlation matrices
 *      3. Computing log-determinants from Cholesky-factorised matrices
 *         as a correctness check (using Equation 4 from the coursework)
 *
 *      All functions are defined as described from the coursework.
 */

#ifndef UTILS_H
#define UTILS_H

#include <cmath>

/**
 * @brief:
 *      Compute a correlation value between two indices as defined from the coursework 
 *      for populating symmetric positive-definite correlation matrices 
 *
 * @param x First index position
 * @param y Second index position
 * @param s Matrix size (s x s)
 *
 * @return:
 *      A double representing the correlation.
 */
inline double corr(double x, double y, double s) {
    return 0.99 * exp(-0.5 * 16.0 * (x - y) * (x - y) / s / s);
}

/**
 * @brief:
 *      Populate a symmetric positive-definite correlation matrix
 *
 *      Fills an n x n matrix stored in row-major order as a 1D flattened array, using the 
 *      correlation function for all (i, j) pairs and overwriting
 *      the diagonal entries in the loop.
 *
 * @param c Pointer to pre-allocated n*n array of doubles (row-major 1D flattened array)
 * @param n Dimension of the square matrix
 *
 * @return:
 *      void: the matrix is written in-place
 */
inline void populate_matrix(double *c, long n) {
    for (long i=0; i<n; i++) {
        for (long j=0; j<n; j++) {
            c[n*i+j] = corr(i, j, n);
        }
        c[n*i+i] = 1.0;
    }
}

/**
 * @brief:
 *      Compute the log-determinant of a matrix from its Cholesky factor
 *
 *      Given a lower-triangular Cholesky factor L (where C = L * L^T),
 *      computes log(det(C)) using Equation 4 from the coursework.
 *
 *      Intended to be called after cholesky_factorisation() has been
 *      applied to the input matrix, and used to compare against the
 *      original matrix as a correctness check.
 *
 * @param m Pointer to the n*n Cholesky factor (row-major 1D flattened array)
 * @param n Dimension of the square matrix
 *
 * @return:
 *      The log-determinant of the original matrix C as a double
 */
inline double compute_log_det(const double *m, long n) {
    double half_log_det = 0.0;
    for (long p=0; p<n; p++) {
        half_log_det += log(m[n*p+p]);
    }
    return 2.0 * half_log_det;
}

#endif // UTILS_H