#!/bin/bash
#!
#! Slurm jobscript for Peta4-IceLake (Ice Lake CPUs, HDR200 IB)
#! Adapted from the example jobscript
#! Cholesky factorisation example using OpenMP

#SBATCH -J cholesky_factorisation_example_icelake_omp
#SBATCH -A MPHIL-DIS-SL2-CPU
#SBATCH -p icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=76
#SBATCH --time=00:10:00
#SBATCH --mail-type=NONE
#SBATCH --output=results/example_icelake_%j.out

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

workdir="$SLURM_SUBMIT_DIR"
cd $workdir
set -e

echo ""
echo "=========================================="
echo "Cholesky Factorisation - Example"
echo "=========================================="
echo "Job ID:              $SLURM_JOB_ID"
echo "Partition:           $SLURM_JOB_PARTITION"
echo "Node:                $SLURM_NODELIST"
echo "CPUs available:      $SLURM_CPUS_ON_NODE"
echo "Date:                $(date)"
echo "=========================================="
export OMP_NUM_THREADS=76
export OMP_SCHEDULE="dynamic,32"

# Small matrix (prints input and output matrices, and log determinant of input matrix)
echo ""
echo "------ Small matrix example (n=5) ------"
./build_icelake/example/example 5

# Larger matrix (print log determinant of input matrix)
echo ""
echo "------ Larger matrix example (n=10000) ------"
./build_icelake/example/example 10000

echo ""
echo "Example completed."