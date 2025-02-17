#!/bin/bash
#SBATCH --job-name=sp_30k_res1
#SBATCH --output=output.txt
#SBATCH --error=error.txt
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
srun job.sh