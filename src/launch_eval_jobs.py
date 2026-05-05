import os
import sys
import time
import subprocess
import argparse
import json

def get_running_job_names():
    try:
        # Run the squeue command for current user
        result = subprocess.run(["squeue", "--me", "--Format=Name:100", "--noheader"],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        # Split the output into lines and strip whitespace
        job_names = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        return job_names
    except subprocess.CalledProcessError as e:
        print(f"Error running squeue: {e.stderr}")
        return []

def create_slurm_scripts(lr,model_name,running_jobs,args,lang,split):
    """Creates a slurm script in right string format

    Args:
        lr (float): learning rate 
        model_name (str): model to train in HF format
        args (argparse.Namespace): args
    Returns:
    - str: the script, will run code evals by default
    """
    if "/" in model_name:
        model_shortname = model_name.split("/")[-1]
    else:
        model_shortname = model_name
    job_name = f"eval-{model_shortname}-lr-{lr}-{lang}-{split}-FSLDR"
    if job_name in running_jobs:
        print(f"Job {job_name} is currently running, skipping...")
        return None
    print(model_shortname)
    output_base_dir = "/scratch/project_2018556/finnish-swedish-long-document-retrieval/results/mteb_evaluations"
    output_folder = os.path.join(output_base_dir,f"{model_shortname}-{lr}")
    final_res = os.path.join(output_folder,f"FinnishSwedishLongDocRetrieval_{lang}_{split}.json")
    print(final_res)
    script_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --account=project_2018556
#SBATCH --time=12:00:00
#SBATCH --nodes=1                   
#SBATCH --ntasks-per-node=1     
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpusmall
#SBATCH --output=../logs/mteb/%x_%j.output
#SBATCH --error=../logs/mteb/%x_%j.error
echo "Start $(date +"%Y-%-m-%d-%H:%M:%S")"
module purge
set -euo pipefail
#Mahti
CONTAINER_PATH="/scratch/project_2017000/containers/pytorch_container_mteb_20260423_221435.sif"
export LOCAL_SCRATCH=/scratch/project_2018556/.cache
export HF_HOME=/scratch/project_2018556/.cache/hf_cache
TMPDIR=$LOCAL_SCRATCH
export MTEB_CACHE=/scratch/project_2018556/.cache/mteb
export HF_HOME=/scratch/project_2018556/.cache/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
export TORCHINDUCTOR_CACHE_DIR="$TMPDIR/torchinductor_cache"
export TRITON_CACHE_DIR="$TMPDIR/triton_cache"
apptainer exec --bind="/users,/projappl,/scratch" --nv $CONTAINER_PATH python run_mteb.py --lr {lr} --model_name {model_name} --lang {lang} --split {split}
echo "End time: $(date)"
""" 
    if os.path.exists(final_res):
        print(f"Model {model_shortname} lr {lr} {split} results exists, skipping...")
        return None
    print(f"Launching job: {job_name}")
    return script_content


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--time',help="time for processing",default="01:00:00")
    ap.add_argument('--dry-run', action='store_true', help="Don't submit any jobs, just print what would be done.")
    ap.add_argument('--test', action='store_true', help="launch one test job")

    running_jobs = get_running_job_names()    
    args = ap.parse_args()
    models = ["finnish-modernbert-large",
            "finnish-modernbert-base",
            "finnish-modernbert-tiny",
            "xlm-roberta-large",
            "mmBERT-base",
            "finnish-modernbert-tiny-short",
            "finnish-modernbert-base-short",
            "finnish-modernbert-large-short",
            "finnish-modernbert-tiny-short-cpt",
            "finnish-modernbert-base-short-cpt",
            "finnish-modernbert-large-short-cpt",
            "finnish-modernbert-tiny-short-edu",
            "finnish-modernbert-large-short-edu",
            "finnish-modernbert-base-short-edu",
            "finnish-modernbert-tiny-edu",
            "finnish-modernbert-large-edu",
            "finnish-modernbert-base-edu"
            ]

    with open("/scratch/project_2018556/finnish-swedish-long-document-retrieval/results/mteb_best_lrs.jsonl") as hp_search_results:
        lines = []
        for l in hp_search_results:
            lines.append(json.loads(l))
        best_lrs = dict([(l['model'],l['lr']) for l in lines if l['model'] in models])

    should_break=False
    dry_run_jobs = 0
    job_count = 0
    

    for m,lr in best_lrs.items():
        if should_break:
            break
        for l in ["fin","swe"]:
            if should_break:
                break
            for s in ['test',"dev"]:
                if should_break:
                    break
                if args.test:
                    if m != "finnish-modernbert-large-short" or s !='test' or l !="fin":
                        continue  
                command = create_slurm_scripts(lr=lr,model_name=m,running_jobs=running_jobs,args=args,lang=l,split=s)
                if command is None:
                    continue
                if args.dry_run:
                    #print(command)
                    dry_run_jobs+=1
                else:
                    temp_file_name = f"{os.getcwd()}/{m}-lr-{lr}_{s}_slurm_job.sh"
                    with open(temp_file_name,"w") as temp_file:
                        temp_file.write(command)
                        # Submit the SLURM job using sbatch with the temporary file
                    result=subprocess.run(["sbatch", temp_file_name], text=True)
                    time.sleep(1)
                    os.remove(temp_file_name)
                    if result.returncode==0:
                        job_count+=1
                    else:
                        print(f"Job submitted successfully: {m}: lr {lr}")
                    running_jobs = get_running_job_names()
                    if args.test:
                        should_break = True
                        break


    print(f"Launched {job_count} jobs")
    if args.dry_run:
        print(f"Would have launched {dry_run_jobs} jobs")

