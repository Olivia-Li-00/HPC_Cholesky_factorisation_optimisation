/**
 * @file:
 *      test_error_cases.cpp
 * @brief:
 *      Error-handling test suite for the Cholesky factorisation implementation
 *
 *      Verifies that the factorisation correctly detects and rejects invalid
 *      inputs by returning -1. Covers 11 test cases across 5 categories:
 *      1. Invalid input arguments: null pointer, n=0, n=-1, n>100,000 (maximum threashold specified in coursework)
 *      2. Non-symmetric input matrices
 *      3. Non-positive-definite matrices: singular, indefinite, negative diagonal
 *      4. Sanity checks: valid symmetric positive definite matrix still succeeds after error cases
 *      5. Multi-threaded error handling: error propagation with multiple threads
 *
 *      Returns 0 if all tests pass, 1 otherwise.
 */
#include "cholesky_factorisation.h"
#include <cmath>
#include <cstdio>
#include <vector>
#include <omp.h> // Needed for setting the number of threads for the multi threads test, using omp_set_num_threads

int main() {

    printf("========================================\n");
    printf("Cholesky factorisation — error case tests\n");
    printf("========================================\n");

    int total_failed_tests = 0;

    // Test 1: Null pointer
    {
        printf("\nTest 1: Null pointer input\n");
        double t = cholesky_factorisation(nullptr, 10);
        if (t < 0.0) {
            printf("[PASS] Returns -1 for null pointer.\n");
        }
        else {
            printf("[FAIL] Did not return -1 for null pointer.\n");
            total_failed_tests++;
        }
    }

    // Test 2: n = 0
    {
        printf("\nTest 2: n=0\n");
        double dummy = 1.0;
        double t = cholesky_factorisation(&dummy, 0);
        if (t < 0.0) {
            printf("[PASS] Returns -1 for n=0.\n");
        }
        else {
            printf("[FAIL] Did not return -1 for n=0.\n");
            total_failed_tests++;
        }
    }

    // Test 3: n = -1
    {
        printf("\nTest 3: n=-1\n");
        double dummy = 1.0;
        double t = cholesky_factorisation(&dummy, -1);
        if (t < 0.0) {
            printf("[PASS] Returns -1 for n=-1.\n");
        }
        else {
            printf("[FAIL] Did not return -1 for n=-1.\n");
            total_failed_tests++;
        }
    }

    // Test 4: n > 100,000
    {
        printf("\nTest 4: n=100001 (exceeds maximum)\n");
        double dummy = 1.0;
        double t = cholesky_factorisation(&dummy, 100001);
        if (t < 0.0) {
            printf("[PASS] Returns -1 for n=100001.\n");
        }
        else {
            printf("[FAIL] Did not return -1 for n=100001.\n");
            total_failed_tests++;
        }
    }

    // Test 5: Non-symmetric matrix (large difference)
    {
        printf("\nTest 5: Non-symmetric matrix\n");

        long n = 3;
        double c[9] = {
            4.0, 2.0, 1.0,
            2.0, 5.0, 3.0,
            1.0, 3.0, 6.0
        };
        c[1] = 999.0;

        double t = cholesky_factorisation(c, n);

        if (t < 0.0) {
            printf("[PASS] Returns -1 for non-symmetric matrix.\n");
        } else {
            printf("[FAIL] Did not return -1 for non-symmetric matrix.\n");
            total_failed_tests++;
        }
    }

    // Test 6: Non-symmetric matrix (subtle difference)
    {
        printf("\nTest 6: Non-symmetric matrix (subtle)\n");

        long n = 3;
        double c[9] = {
            4.0, 2.0, 1.0,
            2.0, 5.0, 3.0,
            1.0, 3.0, 6.0
        };
        c[1] = 2.0 + 1e-5;

        double t = cholesky_factorisation(c, n);

        if (t < 0.0) {
            printf("[PASS] Returns -1 for non-symmetric matrix.\n");
        } else {
            printf("[FAIL] Did not return -1 for non-symmetric matrix.\n");
            total_failed_tests++;
        }
    }

    // Test 7: Singular matrix (det=0)
    {
        printf("\nTest 7: Singular matrix\n");

        long n = 3;
        double c[9] = {
            1.0, 2.0, 3.0,
            2.0, 4.0, 6.0,
            3.0, 6.0, 9.0
        };

        double t = cholesky_factorisation(c, n);

        if (t < 0.0) {
            printf("[PASS] Returns -1 for singular matrix.\n");
        } else {
            printf("[FAIL] Did not return -1 for singular matrix.\n");
            total_failed_tests++;
        }
    }

    // Test 8: Indefinite matrix (has a negative eigenvalue)
    // C = {{1, 2}, {2, 1}} has eigenvalues 3 and -1.
    {
        printf("\nTest 8: Indefinite matrix (negative eigenvalue)\n");

        long n = 2;
        double c[4] = {
            1.0, 2.0,
            2.0, 1.0
        };

        double t = cholesky_factorisation(c, n);

        if (t < 0.0) {
            printf("[PASS] Returns -1 for indefinite matrix.\n");
        } else {
            printf("[FAIL] Did not return -1 for indefinite matrix.\n");
            total_failed_tests++;
        }
    }

    // Test 9: Negative diagonal element
    {
        printf("\nTest 9: Matrix with negative diagonal element\n");

        long n = 2;
        double c[4] = {
           -4.0, 0.0,
            0.0, 1.0
        };

        double t = cholesky_factorisation(c, n);

        if (t < 0.0) {
            printf("[PASS] Returns -1 for matrix with negative diagonal.\n");
        } else {
            printf("[FAIL] Did not return -1 for matrix with negative diagonal.\n");
            total_failed_tests++;
        }
    }

    // Test 10: Valid matrix still works (sanity check)
    // (i.e. error-path tests haven't broken normal operation)
    {
        printf("\nTest 10: Valid symmetric positive-definite matrix still succeeds (sanity check)\n");

        long n = 3;
        double c[9] = {
            4.0, 2.0, 1.0,
            2.0, 5.0, 3.0,
            1.0, 3.0, 6.0
        };

        double t = cholesky_factorisation(c, n);

        if (t >= 0.0) {
            printf("[PASS] Returns >= 0 for valid symmetric positive-definite matrix.\n");
        } else {
            printf("[FAIL] Returned -1 for valid symmetric positive-definite matrix.\n");
            total_failed_tests++;
        }
    }

    // Test 11: Error handling works correctly with multiple threads
    // For single-threaded versions, omp_set_num_threads has no effect;
    // for multi-threaded versions, this verifies that the error flagging
    // mechanism correctly propagates errors when multiple threads are active
    {
        printf("\nTest 11: Error cases with multiple threads\n");
 
        omp_set_num_threads(8); // For setting the number of threads for the multi threads test
 
        // 11a: Non-positive-definite matrix with multiple threads
        {
            long n = 3;
            double c[9] = {
                1.0, 2.0, 3.0,
                2.0, 4.0, 6.0,
                3.0, 6.0, 9.0
            };
 
            double t = cholesky_factorisation(c, n);
 
            if (t < 0.0) {
                printf("[PASS] Returns -1 for singular matrix with 8 threads.\n");
            } else {
                printf("[FAIL] Did not return -1 for singular matrix with 8 threads.\n");
                total_failed_tests++;
            }
        }
 
        // 11b: Valid matrix still works after error case with multiple threads
        {
            long n = 3;
            double c[9] = {
                4.0, 2.0, 1.0,
                2.0, 5.0, 3.0,
                1.0, 3.0, 6.0
            };
 
            double t = cholesky_factorisation(c, n);
 
            if (t >= 0.0) {
                printf("[PASS] Returns >= 0 for valid symmetric positive-definite matrix with 8 threads.\n");
            } else {
                printf("[FAIL] Returned -1 for valid symmetric positive-definite matrix with 8 threads.\n");
                total_failed_tests++;
            }
        }
    }

    // Summary
    printf("\n========================================\n");
    if (total_failed_tests == 0) {
        printf("All error case tests PASSED.\n");
    }
    else {
        printf("%d test(s) FAILED.\n", total_failed_tests);
    }
    printf("========================================\n");

    return (total_failed_tests == 0) ? 0 : 1;
}
