#!/bin/bash
#SBATCH --job-name=Crimes_small_50k
#SBATCH --output=Crimes_small_50k.out
#SBATCH --error=Crimes_small_50k.err
#SBATCH --partition=broadwl
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=36:00:00
#SBATCH --account=pi-ishanu
#SBATCH --mem=20G

module load midway2;
module load python

python /home/azhangs/zed/crimepred_/pycode/api/crimes_small_script.py
