/**
 * @file:
 *      cholesky_factorisation.cpp
 * @brief:
 *      Cache-blocked, multi-threads optimised Cholesky factorisation (v1.2)
 *
 *      This file builds on v1.1 by introducing cache blocking via a three-step
 *      blocked outer loop stepping by block size B along the diagonal:
 *      - Step 1: Factorise the B x B diagonal block (upper triangle only)
 *      - Step 2: Update the panel to the right of the diagonal block
 *        with static scheduling due to uniform workload per column
 *      - Step 3: Trailing submatrix update, with runtime scheduling
 *        via OMP_SCHEDULE to handle flexibility to test for different schedulers due to varied workload
 *
 *      Optimisations carried forward from v1.1:
 *      - OpenMP multi-thread parallelisation
 *      Optimisations carried forward from v0.4:
 *      - SIMD pragmas on inner loops for vectorisation
 *      - __builtin_expect on error path for branch prediction
 *      Optimisations carried forward from v0.3:
 *      - Symmetry exploitation: only upper triangle updated, with final copy pass
 *      Optimisations carried forward from v0.2:
 *      - Multiplication by reciprocal instead of division
 *      Optimisations carried forward from v0.1:
 *      - Reordered submatrix update loops to avoid strided memory access
 *      - Hoisted j-independent elements out of the j-inner loop
 *
 *      Algorithm complexity remains O(n^3).
 */

#include "cholesky_factorisation.h"
#include <cmath>
#include <cstdio>
#include <omp.h>
#include <algorithm>

// Maximum matrix dimension specified in the coursework pdf
#define MAX_N 100000

// Tolerance for symmetry comparison
#define SYMMETRY_TOL 1e-10

// Floating point tolerance
#define FLOAT_TOL 1e-15

// v1.2 Block size for cache blocking
#define B 64

/** Check Symmetry
 *
 * @brief:
 *      Check if a matrix is symmetric within a specified tolerance
 *
 *      This function checks whether a given square matrix is symmetric by comparing
 *      elements c[i,j] and c[j,i] for all i < j. A combined absolute/relative tolerance check is performed.
 *
 * @param c Pointer to the matrix in row-major order (n x n), stored as an 1D array
 * @param n Dimension of the square matrix
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
 *     In-place Cholesky factorisation, cache-blocked multi-threads optimised implementation v1.2
 *
 *           ***** Improvement compared to v1.1 *****
 *     - Introduces cache blocking: the outer loop iterates by block size B,
 *        splitting each block step into three processes (see file header).
 *
 *     1. Mathematical Foundation:
 *         Given any symmetric, positive-definite matrix C, one could factorise such matrix into a
 *         lower-upper form:
 *         C = L L^T
 *         where L is a lower-triangular matrix and L^T is its symmetric transpose
 *     2. Algorithm Description:
 *         This function factorises the C matrix by processing one B x B diagonal block
 *         at a time. Step 1 factorises the block, Step 2 updates the panel to its right, and Step 3
 *         updates the trailing submatrix using all B panel rows. After the blocked loop,
 *         the upper triangle is copied to the lower triangle.
 *     3. Algorithm Complexity:
 *         O(n^3) due to the triple nested loops; improves cache utilisation and runtime.
 *
 * @param c Pointer to a symmetric positive-definite matrix in row-major order (n x n) stored
 *                   as a 1D array. This is modified in-place to store the lower triangular factor L
 *                   and its transpose L^T
 * @param n Dimension of the square matrix (must satisfy 0 < n <= MAX_N)
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

    // v12 Blocked outer loop, with block size B
    for (long p0 = 0; p0 < n; p0 += B) {
        long pb = std::min((long)B, n - p0); // pb: actual block size

        // Step 1: Update the small pb x pb diagonal block (still only the upper triangle)
        for (long k = 0; k < pb; k++) { // k: local index within the current block
            long gk = p0 + k; // gk: global index

            if (__builtin_expect(c[gk*n + gk] < FLOAT_TOL, 0)) {
                fprintf(stderr, "Error: Matrix is not positive-definite at [%ld,%ld] = %.6e.\n", gk, gk, c[gk*n+gk]);
                return -1.0;
            }
            // diagonal update
            double diag = sqrt(c[gk*n + gk]);
            c[gk*n + gk] = diag;
            double inv_diag = 1.0 / diag;

            // Row update
            for (long j = gk+1; j < p0+pb; j++) {
                c[gk*n + j] *= inv_diag;
            }

            // Submatrix update
            for (long i = gk+1; i < p0+pb; i++) {
                double c_pi = c[gk*n + i];
                for (long j = i; j < p0+pb; j++) {
                    c[i*n + j] -= c_pi * c[gk*n + j];
                }
            }
        }

        // precompute inverse diagonals for step 2
        double inv_diags[B];
        for (long k = 0; k < pb; k++) {
            inv_diags[k] = 1.0 / c[(p0+k)*n + (p0+k)];
        }

        // OpenMP multi-thread parallelelisation for step 2 and step 3: higher workload, benefit from parallelisation
        #pragma omp parallel
        {
            // Step 2: Update the panel to the right of the diagonal block
            #pragma omp for schedule(static) // static scheduler can be used, due to uniform workload
            for (long j = p0+pb; j < n; j++) { // iterate each column in the panel
                for (long k = 0; k < pb; k++) { // iterate each row in the panel // k local index
                    long gk = p0 + k; // gk global index

                    c[gk*n + j] *= inv_diags[k]; // firstly being normalised by inverse diagonal
                    double c_gk_j = c[gk*n + j];

                    for (long kk = gk+1; kk < p0+pb; kk++) { // then update rows below it within the same panel column j
                        c[kk*n + j] -= c[gk*n + kk] * c_gk_j;
                    }
                }
            }
            // implicit barrier for all threads at the end of step 2

            // Step 3: Update the trailing submatrix (the submatrix below both the diagonal block and the panel to the right of the diagonal block) (still upper triangle only)
            #pragma omp for schedule(runtime) // scheduler defined by environment variable. If have more time, should use OMP_SCHEDULE environment variable to test which scheduler configuration works best, since there are varied workload
            for (long i = p0+pb; i < n; i++) { // row index

                for (long k = 0; k < pb; k++) {
                    double c_pi = c[(p0+k)*n + i];
                    #pragma omp simd // v04 OpenMP pragma for simd vectorisation
                    for (long j = i; j < n; j++) {
                        c[i*n + j] -= c_pi * c[(p0+k)*n + j];
                    }
                }
            }
            // Implicit barrier for all threads

        } // End of parallel region

    } // End blocked outer loop

    // OpenMP multi-threads parallelisation for the final copy of the upper triangle to the lower triangle directly
    #pragma omp parallel for schedule(runtime) // scheduler defined by environment variable. If have more time, should use OMP_SCHEDULE environment variable to test which scheduler configuration works best
    for (long j = 1; j < n; j++) {
        #pragma omp simd // v04 OpenMP pragma for simd vectorisation
        for (long i = 0; i < j; i++) {
            c[n*j + i] = c[n*i + j];
        }
    }

    // End time measurement and return wall-clock time spent
    double t_end = omp_get_wtime(); // Get another wall-clock timestamp here
    double t_duration = t_end - t_start; // Measure time difference between the 2 wall-clock timestamps
    return t_duration;
}