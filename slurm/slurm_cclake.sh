#!/bin/bash
#!
#! Slurm jobscript for Peta4-CascadeLake (Cascade Lake CPUs, HDR IB)
#! Adapted from the example jobscript
#! Cholesky factorisation testing and benchmark

#SBATCH -J cholesky_factorisation_test_and_benchmark_cclake
#SBATCH -A MPHIL-DIS-SL2-CPU
#SBATCH -p cclake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --mail-type=NONE
#SBATCH --output=results/v00_cclake_%j.out

#! Notes:
#! The Cascade Lake (cclake) nodes have 56 CPUs (cores) each and 3410 MiB of memory per CPU.
#! The --ntasks value refers to the number of tasks to be launched by SLURM only. This
#! usually equates to the number of MPI tasks launched. Reduce this from nodes*56 if
#! demanded by memory requirements, or if OMP_NUM_THREADS>1.
#! Each task is allocated 1 CPU by default, and each CPU is allocated 3420 MiB
#! of memory. If this is insufficient, also specify
#! --cpus-per-task and/or --mem (the latter specifies MiB per node).

#! ####################################################################

. /etc/profile.d/modules.sh
module purge
module load rhel8/default-ccl
module load eigen

workdir="$SLURM_SUBMIT_DIR"

#! safe value to no more than 56:
export OMP_NUM_THREADS=1

cd $workdir
set -e

echo ""
echo "=========================================="
echo "Cholesky factorisation - Cascade Lake"
echo "=========================================="
echo "Job ID:              $SLURM_JOB_ID"
echo "Partition:           $SLURM_JOB_PARTITION"
echo "Node:                $SLURM_NODELIST"
echo "Number of CPUs:      $SLURM_CPUS_ON_NODE"
echo "Number of tasks:     $SLURM_NTASKS"
echo "CPUs per task:       $SLURM_CPUS_PER_TASK"
echo "OMP_NUM_THREADS:     $OMP_NUM_THREADS"
echo "Date:                $(date)"
echo "Working Dir:         $workdir"
echo "=========================================="
echo "Start the program (testing and benchmark)"
echo "=========================================="
echo "=========================================="
echo "Below, errors are expected for testing"
echo "=========================================="
./build_cclake/test/test_error_cases
./build_cclake/test/test_correctness
echo "=========================================="
echo "Performance benchmark"
echo "=========================================="
./build_cclake/benchmark/benchmark
echo ""
echo "Jobscript completed."