import os
os.chdir('/home/azhangs/zed/crimepred_/pycode/api/')

from spatiotemporaldataproc3 import *

proc1 = SpatialTemporalData(path="Crimes2001-2017.csv", bin_path="./bin/procdata")

print("file_dir: {}".format(proc1.file_dir))
print("Now transforming input dataset.")

types1 = ["HOMICIDE", "ASSAULT", "BATTERY"]
TS1 = proc1.transform_with_binary(grid_size=100, type_list=types1, force=True)

proc1.export(path="/home/azhangs/zed/crimepred_/pycode/api/TS1_r2.p")
