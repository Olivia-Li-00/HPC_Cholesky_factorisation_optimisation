/**
 * @file:
 *      cholesky_factorisation.cpp
 * @brief:
 *      Single-thread optimised implementation of Cholesky factorisation algorithm (v0.3)
 *
 *      This file builds on v0.2 with the following optimisations:
 *      - Exploits matrix and algorithm symmetry to halve the work: only update the upper triangle (L^T)
 *        is computed during factorisation, eliminating the separate column update loop and lower triangle of the submatrix update
 *      - The submatrix update inner loop starts at j=i instead of j=p+1, updating
 *        only the upper triangle of the submatrix
 *      - A final copy pass writes the upper triangle to the lower triangle,
 *        with j-outer/i-inner ordering to prioritise contiguous writes over reads
 *
 *      Optimisations carried forward from v0.2:
 *      - Multiplication by reciprocal instead of division
 *      Optimisations carried forward from v0.1:
 *      - Reordered submatrix update loops to avoid strided memory access
 *      - Hoisted j-independent elements out of the j-inner loop
 *
 *      All other aspects (input validation, in-place computation, timing) remain
 *      identical to v0.0. The algorithm has overall O(n^3) complexity.
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
 *     In-place Cholesky factorisation, optimised implementation v0.3
 *
 *           ***** Improvement compared to v0.2 *****
 *     - Exploits symmetry of the input matrix: only computes and updates the
 *       upper triangle (L^T) during factorisation, removing the separate column
 *       normalisation loop entirely and the update of the lower triangle of the submatrix update
 *     - The hoisted variable changes from c[n*i+p] (column element) to c[n*p+i]
 *       (row element), since only the upper triangle is maintained
 *     - The submatrix inner loop range changes from j=p+1..n to j=i..n,
 *       skipping the lower triangle and roughly halving the update work
 *     - A separate copy pass at the end fills the lower triangle (L) from
 *       the upper triangle (L^T), using j-outer/i-inner ordering to prioritise
 *       contiguous writes over contiguous reads
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
 *         - Normalise the row by this diagonal value (upper triangle only)
 *         - Update the remaining upper-triangular submatrix using outer product elimination
 *         After the main loop, copy the upper triangle to the lower triangle
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

    // Cholesky factorisation
    for (long p=0; p<n; p++) { // Move along the diagonal of the matrix

        // Checks for input requirements
        // 4) Input matrix must be positive-definite
        if (c[p*n+p] < FLOAT_TOL) {
            fprintf(stderr, "Error: Matrix is not positive-definite, diagonal elements at [i,j]=[%ld,%ld] is %.6e.\n", p, p, c[p*n+p]);
            return -1.0;
        }

        // v02 multiplication instead of division
        double diag=sqrt(c[p*n+p]);
        c[n*p+p]=diag; // Update diagonal element
        double inv_diag = 1.0 / diag;

        // v03 only keep the row update and remove the column update due to symmetry
        // v03 only need to update on one side, choose row over column, for contiguous access
        for (long j=p+1; j<n; j++) { // Update row to right of diagonal element
            c[n*p+j]*=inv_diag; // v02 multiplication instead of division
        }

        // v03 only need to update the upper triangle (L^T) due to symmetry
        // v03 this means need to update c[n*i+j] using c[n*p+i] not c[n*i+p], hence change the hoisting term
        // v01 Avoid strided read and write
        for (long i=p+1; i<n; i++) { // Update submatrix below-right of the diagonal element
            // v01 Hoisting
            double c_pi = c[n*p+i];
            for (long j=i; j<n; j++) {
                c[n*i+j] -= c_pi * c[n*p+j];
            }
        }
    }

    // v03 copy the upper triangle (L^T) to lower triangle (L) directly
    // v03 j is the outer loop and i is the inner loop: prioritise contiguous write over contiguous read, due to the cost of cache coherence
    for (long j=1; j<n; j++) {
        for (long i=0; i<j; i++) {
            c[n*j+i] = c[n*i+j];
        }
    }

    // End time measurement and return wall-clock time spent
    double t_end = omp_get_wtime(); // Get another wall-clock timestamp here
    double t_duration = t_end - t_start; // Measure time difference between the 2 wall-clock timestamps
    return t_duration;
}