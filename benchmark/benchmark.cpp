/**
 * @file:
 *      benchmark.cpp
 * @brief:
 *      Benchmarking suite for Cholesky factorisation implementations
 *
 *      This file provides a comprehensive benchmarking framework to evaluate the
 *      performance of Cholesky factorisation implementations across different
 *      matrix dimensions. The benchmark:
 *      - Measures execution time using OpenMP timestamps
 *      - Computes statistical metrics (mean and standard deviation) across multiple runs
 *      - Supports both predefined matrix sizes and custom sizes via environment variables
 *
 *      The benchmarking methodology follows scientific best practices by:
 *      - Performing multiple runs per configuration to account for system variability
 *      - Applying Bessel's correction for unbiased standard deviation estimation
 *      - Reporting timer precision to contextualise measurement accuracy
 */

#include "cholesky_factorisation.h"
#include "utils.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <omp.h>

#define NUM_REPEATS 5

const long SIZES[] = {100, 500, 1000, 2000};
const int NUM_SIZES = sizeof(SIZES) / sizeof(SIZES[0]);

/**
 * @brief:
 *      Run benchmark for a specific matrix size configuration
 *
 *      Executes multiple repetitions of Cholesky factorisation for a given matrix size,
 *      computing statistical metrics (average and standard deviation) for performance evaluation.
 *      For each repetition:
 *      1. Generates a symmetric positive-definite test matrix
 *      2. Performs factorisation and validates success
 *      3. Records execution time
 *
 * @param                        long n: Dimension of the square matrix to benchmark
 * @param                   int repeats: Number of repetitions to perform (must be > 1)
 * @param    double *avg_across_repeats: Output parameter for average execution time
 * @param double *stddev_across_repeats: Output parameter for standard deviation
 *
 * @return:
 *      0 on success, -1 on error (invalid parameters or factorisation failure)
 *
 * Note: Uses Bessel's correction (N-1) for unbiased standard deviation estimation
 */
int benchmark_individual(long n, int repeats, double *avg_across_repeats, double *stddev_across_repeats) {
    
    // Checks: the number of repeats (of experiments of the same matrix size) must be > 1
    if (repeats <= 1) {
        fprintf(stderr, "Error: Number of repeats must be > 1, currently %d.\n", repeats);
        return -1;
    }

    std::vector<double> runs(repeats);

    for (int run = 0; run < repeats; run++) {
        std::vector<double> c(n*n);
        populate_matrix(c.data(), n);
        double t = cholesky_factorisation(c.data(), n);
        if (t < 0.0) {
            fprintf(stderr, "Error: Factorisation failed.\n");
            return -1;
        }
        runs[run] = t;
    }

    // Compute the average time across multiple runs of the same config
    double sum = 0.0;
    for (int run = 0; run < repeats; run++) {
        sum += runs[run];
    }
    double avg = sum / repeats;

    // Compute the standard deviation across multiple runs of the same config
    double var_sum = 0.0;
    for (int run = 0; run < repeats; run++) {
        double diff = runs[run] - avg;
        var_sum += diff * diff;
    }
    double stddev = sqrt(var_sum / (repeats - 1)); // Use the N-1 formula to get unbiased std estimate with Bessel correction

    *avg_across_repeats = avg;
    *stddev_across_repeats = stddev;

    return 0;
}

/**
 * @brief:
 *      Main entry point for the benchmark
 *
 *      Implements two operational modes based on environment configuration:
 *      1. Single-size mode: When SINGLE_N environment variable is set in the slurm script or via command line,
 *         benchmarks only that specific matrix dimension
 *      2. Multi-size mode: When no environment variable is set, benchmarks all sizes
 *         defined in the SIZES array at the top of this file
 *
 *      For each configuration, the benchmark:
 *      - Reports OpenMP timer precision for measurement context
 *      - Outputs results with average time and standard deviation
 *      - Validates successful factorisation before including timing in results
 *
 * @return:
 *      0 on successful benchmark completion, 1 on error conditions
 */
int main() {

    // This openMP function reports the resolution of the omp_get_wtime() measurements, by reporting the time difference between 2 consecutive ticks of the clock
    // This is needed to interpret the time measurement results, i.e., to know the precision of the timer itself
    printf("OpenMP timer precision: %.3e seconds.\n", omp_get_wtick());

    // Option 1: Using environmental variable which specified a specific single n, needed for later versions
    const char *single_n_env = getenv("SINGLE_N");
    if (single_n_env) {
        long n = atol(single_n_env);
        printf("For a specific matrix size: n=%ld\n", n);
        printf("The number of repeats per configuration: %d repeats.\n\n", NUM_REPEATS);
        printf("%-10s  %-20s  %-20s\n", "n", "Average time(s)", "Standard deviation(s)");
        printf("-----------------------------------------------------\n");
        double avg, stddev;
        if (benchmark_individual(n, NUM_REPEATS, &avg, &stddev) != 0) {
            printf("%-10ld  ERROR\n", n);
            return 1;
        }
        printf("%-10ld  %-20.6f  %-20.6f\n", n, avg, stddev);
    } 
    else {
        // Option 2: Looping over each of the size (n) specified in SIZES
        printf("The number of repeats per configuration (different matrix sizes): %d repeats.\n\n", NUM_REPEATS);
        printf("%-10s  %-20s  %-20s\n", "n", "Average time(s)", "Standard deviation(s)");
        printf("-----------------------------------------------------\n");
        for (int size = 0; size < NUM_SIZES; size++) {
            long n = SIZES[size];
            double avg, stddev;
            if (benchmark_individual(n, NUM_REPEATS, &avg, &stddev) != 0) {
                printf("%-10ld  ERROR\n", n);
                return 1;
            }
            printf("%-10ld  %-20.6f  %-20.6f\n", n, avg, stddev);
        }
    }

    printf("\nBenchmark Complete.\n");

    return 0;
}
