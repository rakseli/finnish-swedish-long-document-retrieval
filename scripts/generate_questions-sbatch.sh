#!/bin/bash
#SBATCH --job-name=generate_questions
#SBATCH --account=project_2018556
#SBATCH --time=03:00:00
#SBATCH --nodes=1                   
#SBATCH --ntasks-per-node=1     
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:a100:4
#SBATCH --partition=gpumedium
#SBATCH --output=../logs/quetion_generation/%x_%j.output
#SBATCH --error=../logs/quetion_generation/%x_%j.error
echo "Start $(date +"%Y-%-m-%d-%H:%M:%S")"
module purge
set -euo pipefail
#Mahti
CONTAINER_PATH="/scratch/project_2017000/containers/pytorch_container_20260407_205334.sif"
export LOCAL_SCRATCH=/scratch/project_2018556/.cache
TMPDIR=$LOCAL_SCRATCH
echo "Cache location: $TMPDIR"
#Python
export PYTHONWARNINGS=ignore
#vllm and torch
export VLLM_CONFIGURE_LOGGING=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_CACHE_ROOT="$TMPDIR/vllm_cache"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
export TORCHINDUCTOR_CACHE_DIR="$TMPDIR/torchinductor_cache"
export TRITON_CACHE_DIR="$TMPDIR/triton_cache"
#DISTRIBUTED
export OMP_NUM_THREADS=1
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=9999
apptainer exec --bind="/users,/projappl,/scratch,$LOCAL_SCRATCH" --nv $CONTAINER_PATH python ../src/generate_questions.py --lang swe --exit_duration_in_mins 175
echo "End $(date +"%Y-%-m-%d-%H:%M:%S")"