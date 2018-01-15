#!/bin/bash
#SBATCH --job-name=ts1
#SBATCH --output=ts1.out
#SBATCH --error=ts1.err
#SBATCH --partition=broadwl
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=20:00:00
#SBATCH --account=pi-ishanu
#SBATCH --mem=20G

module load midway2;
module load python

python /home/azhangs/zed/crimepred_/pycode/api/ts1_script.py
