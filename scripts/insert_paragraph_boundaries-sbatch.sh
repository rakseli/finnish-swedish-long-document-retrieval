#!/bin/bash
#SBATCH --job-name=test_vllm
#SBATCH --account=project_2018556
#SBATCH --time=00:05:00
#SBATCH --nodes=1                   
#SBATCH --ntasks-per-node=1     
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:a100:2
#SBATCH --partition=gputest
#SBATCH --output=../logs/preprocessing/%x_%j.output
#SBATCH --error=../logs/preprocessing/%x_%j.error
echo "Start $(date +"%Y-%-m-%d-%H:%M:%S")"
module purge
module load pytorch/2.9
set -euo pipefail
#Mahti
export LOCAL_SCRATCH=/scratch/project_2018556/.cache
TMPDIR=$LOCAL_SCRATCH
echo "Cache location: $TMPDIR"
#Python
export PYTHONWARNINGS=ignore
#vllm and torch
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_LOG_LEVEL=DEBUG
export VLLM_CACHE_ROOT="$TMPDIR/vllm_cache"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
export TORCHINDUCTOR_CACHE_DIR="$TMPDIR/torch_inductor_cache"
export TRITON_CACHE_DIR="$TMPDIR/triton_cache"
#DISTRIBUTED
export OMP_NUM_THREADS=1
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=9999
srun python ../src/insert_paragraph_boundaries.py --test