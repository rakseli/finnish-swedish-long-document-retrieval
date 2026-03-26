#!/bin/bash
#SBATCH --job-name=create_chunked_data
#SBATCH --account=project_2018556
#SBATCH --time=03:00:00
#SBATCH --nodes=1                   
#SBATCH --ntasks-per-node=1     
#SBATCH --cpus-per-task=1
#SBATCH --partition=small
#SBATCH --output=../logs/preprocessing/%x_%j.output
#SBATCH --error=../logs/preprocessing/%x_%j.error

set -euo pipefail
echo "Start $(date +"%Y-%m-%d-%H:%M:%S")"

module load python-data
INPUT_PATH=$1
echo "Using input path: $INPUT_PATH"
srun python ../src/create_chunked_data.py --input_path $INPUT_PATH
echo "End $(date +"%Y-%m-%d-%H:%M:%S")"