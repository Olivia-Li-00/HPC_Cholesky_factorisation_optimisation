/**
 * @file:
 *      cholesky_factorisation.cpp
 * @brief:
 *      Multi-thread optimised implementation of Cholesky factorisation algorithm (v1.1)
 *
 *      This file builds on v0.4 by introducing OpenMP multi-thread parallelisation:
 *      - A single #pragma omp parallel region wraps the entire factorisation
 *      - The diagonal update is executed by one thread via #pragma omp single
 *      - The row normalisation loop is parallelised with #pragma omp for simd
 *        using static scheduling (uniform workload per iteration)
 *      - The submatrix update loop is parallelised with #pragma omp for using
 *        runtime-configurable scheduling via OMP_SCHEDULE
 *      - The symmetry copy is parallelised with #pragma omp for using
 *        runtime-configurable scheduling via OMP_SCHEDULE
 *      - A shared error_flag replaces direct return for positive-definiteness
 *        failures, since return is not permitted inside a parallel region
 *
 *      Optimisations carried forward from v0.4:
 *      - SIMD pragmas on inner loops for vectorisation and __builtin_expect on error path to help branch prediction
 *      Optimisations carried forward from v0.3:
 *      - Symmetry exploitation: only upper triangle updated, with final copy pass
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
#include <omp.h>

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
 *     In-place Cholesky factorisation, multi-thread optimised implementation v1.1
 *
 *           ***** Improvement compared to v0.4 *****
 *     - Introduces OpenMP multi-thread parallelisation within a single
 *       parallel region that persists across all diagonal steps
 *     - The diagonal element update (sqrt, reciprocal) is executed by a
 *       single thread via #pragma omp single; the implicit barrier ensures
 *       all threads see the updated values before proceeding
 *     - The row normalisation is distributed across threads with
 *       #pragma omp for simd schedule(static), combining SIMD (data-level) and thread-level 
 *       parallelism with static scheduling due to uniform workload
 *     - The submatrix update is distributed with #pragma omp for
 *       schedule(runtime), allowing the schedule to be tuned at
 *       runtime via the OMP_SCHEDULE environment variable
 *     - The symmetry copy is similarly parallelised with #pragma omp for
 *       schedule(runtime) with runtime scheduling via environment variable
 *     - A shared error_flag variable replaces direct return -1.0 for
 *       non-positive-definite detection, since return statements are not
 *       permitted inside OpenMP parallel regions
 *
 *     1. Mathematical Foundation:
 *         Given any symmetric, positive-definite matrix C, one could factorise such matrix into a
 *         lower-upper form:
 *         C = L L^T
 *         where L is a lower-triangular matrix and L^T is its symmetric transpose
 *     2. Algorithm Description:
 *         This function effectively factorises the C matrix by processing one diagonal element at a time.
 *         For each diagonal position p:
 *         - Compute sqrt(C[p,p]) as the new diagonal value (single thread)
 *         - Normalise the row by this diagonal value (upper triangle only, multi threads)
 *         - Update the remaining upper-triangular submatrix using outer product elimination (multi threads)
 *         After the main loop, copy the upper triangle to the lower triangle (multi-threads)
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

    // v11 shared variable
    double diag;
    double inv_diag;
    // v11 shared error flag
    int error_flag = 0;

    // Start time measurement (using omp throughout the project for consistency)
    double t_start = omp_get_wtime(); // Get a wall-clock timestamp

    // v11 OpenMP multi-thread parallelisation
    #pragma omp parallel
    {
        // Cholesky factorisation
        for (long p=0; p<n; p++) { // Move along the diagonal of the matrix

            // v11 only one thread executes each diagonal update
            #pragma omp single
            {
                // v11 using error_flag since can't use return -1.0
                // Checks for input requirements
                // 4) Input matrix must be positive-definite
                // v04 to help with branch prediction: use __builtin_expect with argument 0 (the input will likely to be a positive definite matrix)
                if (__builtin_expect(c[p*n+p] < FLOAT_TOL, 0)) {
                    fprintf(stderr, "Error: Matrix is not positive-definite, diagonal elements at [i,j]=[%ld,%ld] is %.6e.\n", p, p, c[p*n+p]);
                    error_flag = 1;
                }
                if (!error_flag) {
                    // v02 multiplication instead of division
                    diag=sqrt(c[p*n+p]);
                    c[n*p+p]=diag; // Update diagonal element
                    inv_diag = 1.0 / diag;
                }
            }
            if (error_flag) break;

            // v11 OpenMP for loop together with simd
            // v04 OpenMP pragma for simd vectorisation
            // v03 only keep the row update and remove the column update due to symmetry
            // v03 only need to update on one side, choose row over column, for contiguous access
            #pragma omp for simd schedule(static) // static clause is used, as workload per iteration is the same
            for (long j=p+1; j<n; j++) { // Update row to right of diagonal element
                c[n*p+j]*=inv_diag; // v02 multiplication instead of division
            }
            #pragma omp barrier // Barrier for all threads at the row update, implict barrier probably already exists, but I implement it explictly just in case

            // v11 OpenMP for loop
            // v03 only need to update the upper triangle (L^T) due to symmetry
            // v03 this means need to update c[n*i+j] using c[n*p+i] not c[n*i+p], hence change the hoisting term
            // v01 Avoid strided read and write
            #pragma omp for schedule(runtime) // scheduling will be set in the environment variable and will be tested, since there are varied workloads
            for (long i=p+1; i<n; i++) { // Update submatrix below-right of the diagonal element
                // v01 Hoisting
                double c_pi = c[n*p+i];
                // v04 OpenMP pragma for simd vectorisation
                #pragma omp simd
                for (long j=i; j<n; j++) {
                    c[n*i+j] -= c_pi * c[n*p+j];
                }
            }
            #pragma omp barrier // Barrier for all threads at the submatrix update, implict barrier probably already exists, but I implement it explictly just in case
        }
        
        // v11 Didn't add if (!error_flag) condition here to optimise the runtime performance, if the error_flag = 1, then it the function will return -1.0 at the end anyway
        // v11 OpenMP for loop
        // v03 copy the upper triangle (L^T) to lower triangle (L) directly
        // v03 j is the outer loop and i is the inner loop: prioritise contiguous write over contiguous read, due to the cost of cache coherence
        #pragma omp for schedule(runtime) // scheduling will be set in the environment variable and will be tested, since there are varied workloads
        for (long j=1; j<n; j++) {
            // v04 OpenMP pragma for simd vectorisation
            #pragma omp simd
            for (long i=0; i<j; i++) {
                c[n*j+i] = c[n*i+j];
            }
        }
        #pragma omp barrier // Barrier for all threads for the symmetry copy step, implict barrier probably already exists, but I implement it explictly just in case
    } // v11 OpenMP multi-threads parallel region ends here

    // v11 if the matrix is not positive definite, return -1.0
    if (error_flag) return -1.0;


    // End time measurement and return wall-clock time spent
    double t_end = omp_get_wtime(); // Get another wall-clock timestamp here
    double t_duration = t_end - t_start; // Measure time difference between the 2 wall-clock timestamps
    return t_duration;
}