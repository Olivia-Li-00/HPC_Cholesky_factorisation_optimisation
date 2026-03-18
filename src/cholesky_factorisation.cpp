/**
 * @file:
 *      cholesky_factorisation.cpp
 * @brief:
 *      Baseline implementation of Cholesky factorisation algorithm (v0.0)
 *
 *      This file provides a reference implementation of the Cholesky factorisation
 *      for symmetric positive-definite matrices. The implementation:
 *      - Performs comprehensive input validation (matrix size, symmetry, positive-definiteness)
 *      - Uses in-place computation to factor matrix C into L L^T form
 *      - Measures execution time using omp_get_wtime for performance evaluation
 *      - Serves as the baseline version against which optimisations will be compared
 *
 *      The algorithm has overall O(n^3) complexity.
 */

#include "cholesky_factorisation.h"
#include <cmath>
#include <cstdio>
#include <omp.h> // Using its omp_get_wtime function for wall time measurement consistently,
                 // so that same method is used throughout the project development

// Maximum matrix dimension specified in the coursework pdf
#define MAX_N 100000

// Tolerance for symmetry comparison
#define SYMMETRY_TOL 1e-10

// Floating point tolerance
#define FLOAT_TOL 1e-15

/** Check Symmetry
 *
 * @brief:
 *      Check if a matrix is symmetric within a specified tolerance
 *
 *      This function checks whether a given square matrix is symmetric by comparing
 *      elements c[i,j] and c[j,i] for all i < j. A combined absolute/relative tolerance check is performed.
 *
 * @param const double *c: Pointer to the matrix in row-major order (n x n), stored as an 1D array
 * @param    const long n: Dimension of the square matrix
 *
 * @return:  0 if the matrix is symmetric within tolerance 
 *          -1 otherwise
 */
static int check_symmetry(const double *c, const long n) {
    for (long i=0; i<n; i++) {
        for (long j=i+1; j<n; j++) {
            double diff = fabs(c[n*i+j]-c[n*j+i]);
            double scale = fabs(c[n*i+j]) + fabs(c[n*j+i]);
            if (diff > SYMMETRY_TOL * (1 + scale)) { // Use relative tolerance for larger elements, absolute tolerance for small elements
                return -1;
            }
        }
    }
    return 0;
}

/** Cholesky Factorisation Algorithm
 *
 * @brief: 
 *     In-place Cholesky factorisation, baseline implementation v0.0 
 *
 *     1. Mathematical Foundation:
 *         Given any symmetric, positive-definite matrix C, one could factorise such matrix into a
 *         lower-upper form:
 *         C = L L^T
 *         where L is a lower-triangular matrix and L^T is its symmetric transpose
 *     2. Algorithm Description:
 *         This function effectively factorises the C matrix by processing one diagonal element at a time.
 *         For each diagonal position p:
 *         - Compute sqrt(C[p,p]) as the new diagonal value
 *         - Normalise the row/column by this diagonal value
 *         - Update the remaining submatrix using outer product elimination
 *     3. Algorithm Complexity:
 *         O(n^3) due to the triple nested loops processing n diagonal elements with O(n^2) operations each
 *
 * @param double *c: Pointer to a symmetric positive-definite matrix in row-major order (n x n) stored
 *                   as a 1D array. This is modified in-place to store the lower triangular factor L 
 *                   and its transpose L^T
 * @param    long n: Dimension of the square matrix (must satisfy 0 < n <= MAX_N)
 *
 * @return: On success: Wall-clock time (seconds) taken for factorisation
 *          On failure: -1.0 with appropriate error message printed to stderr
 */
double cholesky_factorisation(double *__restrict__ c, const long n) {
    
    // Checks for input requirements
    // 1) Successful memory allocation and initialisation of matrix c
    // 2) Valid matrix dimension requirement from the coursework: N <= 100,000 and natural requirement N > 0
    // 3) Input matrix must be symmetric

    if (c == nullptr) {
        fprintf(stderr, "Error: Null pointer passed to cholesky_factorisation, check the memory allocation and initialisation of input matrix.\n");
        return -1.0;
    }

    if (n <= 0) {
        fprintf(stderr, "Error: Invalid input matrix dimension n=%ld (must be > 0).\n", n);
        return -1.0;
    }

    if (n > MAX_N) {
        fprintf(stderr, "Error: Invalid input matrix dimension n=%ld (exceeds maximum allowed %d).\n", n, MAX_N);
        return -1.0;
    }

    if (check_symmetry(c, n) != 0) {
        fprintf(stderr, "Error: Input matrix is not symmetric.\n");
        return -1.0;
    }

    // Start time measurement (using omp throughout the project for consistency)
    double t_start = omp_get_wtime(); // Get a wall-clock timestamp

    // Cholesky factorisation: main step
    for (long p=0; p<n; p++) { // Move along the diagonal of the matrix

        // Checks for input requirements
        // 4) Input matrix must be positive-definite
        if (c[p*n+p] < FLOAT_TOL) {
            fprintf(stderr, "Error: Matrix is not positive-definite, diagonal elements at [i,j]=[%ld,%ld] is %.6e.\n", p, p, c[p*n+p]);
            return -1.0;
        }

        double diag=sqrt(c[p*n+p]);
        c[n*p+p]=diag; // Update diagonal element

        for (long j=p+1; j<n; j++) { // Update row to right of diagonal element
            c[n*p+j]/=diag;
        }

        for (long i=p+1; i<n; i++) { // Update column below diagonal element
            c[n*i+p]/=diag;
        }

        for (long j=p+1; j<n; j++) { // Update submatrix below-right of the diagonal element
            for (long i=p+1; i<n; i++) {
                c[n*i+j] -= c[n*i+p]*c[n*p+j];
            }
        }
    }

    // End time measurement and return wall-clock time spent
    double t_end = omp_get_wtime(); // Get another wall-clock timestamp here
    double t_duration = t_end - t_start; // Measure time difference between the 2 wall-clock timestamps
    return t_duration;
}
