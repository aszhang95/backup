import os
os.chdir('/home/azhangs/zed/crimepred_/pycode/api/')

from spatiotemporaldataproc3 import *

proc2 = SpatialTemporalData(path="Crimes2001-2017.csv", bin_path="./bin/procdata")

print("file_dir: {}".format(proc2.file_dir))
print("Now transforming input dataset.")

types2 = ["THEFT", "MOTOR VEHICLE THEFT", "ROBBERY", "BURGLARY"]
TS2 = proc2.transform_with_binary(grid_size=100, type_list=types2, force=True)

proc2.export(path="/home/azhangs/zed/crimepred_/pycode/api/TS2_r2.p")
