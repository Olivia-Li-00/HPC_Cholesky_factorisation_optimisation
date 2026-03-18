#!/bin/bash
#!
#! Slurm jobscript for Peta4-IceLake (Ice Lake CPUs, HDR200 IB)
#! Adapted from the example jobscript (Last updated: Sat Jul 31 15:39:45 BST 2021)
#! Cholesky factorisation testing and benchmark

#SBATCH -J cholesky_factorisation_test_and_benchmark_icelake
#SBATCH -A MPHIL-DIS-SL2-CPU
#SBATCH -p icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00
#SBATCH --mail-type=NONE
#SBATCH --output=results/v02_icelake_%j.out

#! Notes:
#! The Ice Lake (icelake) nodes have 76 CPUs (cores) each and 3380 MiB of memory per CPU.
#! The --ntasks value refers to the number of tasks to be launched by SLURM only. This
#! usually equates to the number of MPI tasks launched. Reduce this from nodes*76 if
#! demanded by memory requirements, or if OMP_NUM_THREADS>1.
#! Each task is allocated 1 CPU by default, and each CPU is allocated 3380 MiB
#! of memory. If this is insufficient, also specify
#! --cpus-per-task and/or --mem (the latter specifies MiB per node).

#! ####################################################################

. /etc/profile.d/modules.sh
module purge
module load rhel8/default-icl
module load eigen

workdir="$SLURM_SUBMIT_DIR"

#! safe value to no more than 76:
export OMP_NUM_THREADS=1

cd $workdir
set -e

echo ""
echo "=========================================="
echo "Cholesky factorisation - Icelake"
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
./build_icelake/test/test_error_cases
./build_icelake/test/test_correctness
echo "=========================================="
echo "Performance benchmark"
echo "=========================================="
./build_icelake/benchmark/benchmark
echo ""
echo "Jobscript completed."