# C2 High Performance Computing Coursework

This coursework implemented an in-place Cholesky factorisation library, optimised with multi-threaded parallelisations via OpenMP on CSD3.

## Directory Structure

- `src/` Source code
- `include/` Headers
- `example/` Usage example
- `test/` Test suites
- `benchmark/` Benchmarking suite
- `slurm/` CSD3 submission scripts
- `results/` Benchmark outputs
- `report/` Report, figures, and Jupyter notebook for visualisation
- `build_doxygen/` Automatic documentation

## Usage Instructions

On CSD3 Ice Lake Partition:

```bash
module purge && module load rhel8/default-icl && module load eigen
cmake -B build_icelake -DCMAKE_CXX_COMPILER=icpx -DCMAKE_CXX_FLAGS="-O3 -xHost"
cmake --build build_icelake
```
Replace -xHost with `-march=native` if using different compiler.

## Usage Examples

```cpp
#include "cholesky_factorisation.h"

...

std::vector<double> c(n * n);
populate_matrix(c.data(), n);
double time = cholesky_factorisation(c.data(), n);
printf("Factorisation completed in %.6f seconds.\n", time);
```
See `example/example.cpp` for a complete working example. Example CSD3 submission scripts are in `slurm/` (e.g. `sbatch slurm/slurm_icelake_example.sh` used for examples, `sbatch slurm/slurm_icelake_optimal.sh` used in every version, `sbatch slurm/slurm_icelake_test.sh` used for v1.1 tuning).

## Performance Guidance

 In general, multi-threading is beneficial for n >= ~3000; for smaller matrices, single-threaded execution avoids thread overhead. Threads number can be set via `OMP_NUM_THREADS` and schedule configurations can be set via`OMP_SCHEDULE`.

## Automatic Documentation with Doxygen

```bash
cmake --build build_doxygen -t doxygen
cmake -B build_doxygen
# output in build_doxygen/html/
```

## Version History

| Tag | Description |
|-----|-------------|
| v0.0 | Baseline with input validation |
| v0.1 | Loop reorder for contiguous access + hoisting |
| v0.2 | Reciprocal multiplication |
| v0.3 | Symmetry |
| v0.4 | SIMD pragmas + branch hint |
| v1.1 | OpenMP multi-threaded parallelisation |
| v1.2 | Cache-blocked parallelisation (B=64) |

See `report/report.pdf` for detailed performance analysis.

## Declaration of AI tool usage

AI chatbox has been used to discuss mathematical and scientific background, consultation on use of specific tools including C++ standard libraries.