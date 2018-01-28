#!/bin/bash
#SBATCH --job-name=ts2
#SBATCH --output=ts2.out
#SBATCH --error=ts2.err
#SBATCH --partition=broadwl
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=36:00:00
#SBATCH --account=pi-ishanu
#SBATCH --mem=20G

module load midway2;
module load python

python /home/azhangs/zed/crimepred_/pycode/api/ts2_script.py
