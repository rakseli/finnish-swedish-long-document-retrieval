#!/bin/bash
#SBATCH --job-name=filter_corpus
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
INPUT_ROOT=$1
echo "Using input root: $INPUT_ROOT"
srun python /scratch/project_2018556/finnish-swedish-long-document-retrieval/src/filter_corpus.py --input_root $INPUT_ROOT
echo "End $(date +"%Y-%m-%d-%H:%M:%S")"