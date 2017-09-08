# midway2 commands to know:

# instantiating cluster on midway2
sinteractive --exclusive --partition=broadwl --time=5:00:00

# constricting computational power on login node to not get kicked off
export OMP_NUM_THREADS=10
