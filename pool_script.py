import os
os.chdir('/home/azhangs/zed/crimepred_/pycode/api/')

from stdp6_pool import *

proc1 = SpatialTemporalData(path="Crimes_small_50k.csv", bin_path="./bin/procdata")

print("file_dir: {}".format(proc1.file_dir))
print("Now transforming input dataset.")

types1 = ["HOMICIDE", "ASSAULT", "BATTERY"]
TS1 = proc1.transform(grid_size=100, type_list=types1, force=True, out_fname="TS1_pool_test.csv")

proc1.export(path="/home/azhangs/zed/crimepred_/pycode/api/TS1_pool_test.p")
