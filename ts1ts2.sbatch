#!/bin/bash
#SBATCH --job-name=ts1ts2
#SBATCH --output=ts1ts2.out
#SBATCH --error=ts1ts2.err
#SBATCH --partition=broadwl
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=36:00:00
#SBATCH --account=pi-ishanu
#SBATCH --mem=20G

module load midway2;
module load python

python /home/azhangs/zed/crimepred_/pycode/api/ts1ts2_script.py
