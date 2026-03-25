#!/bin/bash
#SBATCH --job-name=count_tokens
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
OUTPUT_PATH=$2
echo "Using input file: $INPUT_PATH"
echo "Using output file: $OUTPUT_PATH"
srun python /scratch/project_2018556/finnish-swedish-long-document-retrieval/src/count_tokens.py --input_file $INPUT_PATH --output_file $OUTPUT_PATH

echo "End $(date +"%Y-%m-%d-%H:%M:%S")"