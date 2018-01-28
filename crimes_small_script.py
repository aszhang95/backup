import os
os.chdir('/home/azhangs/zed/crimepred_/pycode/api/')

from spatiotemporaldataproc3 import *

proc = SpatialTemporalData(path="Crimes_small_50k.csv", bin_path="./bin/procdata")

print("file_dir: {}".format(proc.file_dir))
print("Now transforming input dataset.")

types = ["HOMICIDE", "ASSAULT", "BATTERY"]
TS_small = proc.transform_with_binary(grid_size=100, type_list=types, force=True)
print(TS_small)

proc.export(path="/home/azhangs/zed/crimepred_/pycode/api/TS_small_out.p")
