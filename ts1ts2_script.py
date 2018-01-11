import os
os.chdir('/home/azhangs/zed/crimepred_/pycode/api/')

from spatiotemporaldataproc3 import *

proc1 = SpatialTemporalData(path="Crimes2001-2017.csv", bin_path="./bin/procdata")
proc2 = SpatialTemporalData(path="Crimes2001-2017.csv", bin_path="./bin/procdata")

types1 = ["HOMICIDE", "ASSAULT", "BATTERY"]
types2 = ["THEFT", "MOTOR VEHICLE THEFT", "ROBBERY", "BURGLARY"]

TS1 = proc1.transform_with_binary(grid_size=100, type_list=types1, force=True)
TS2 = proc2.transform_with_binary(grid_size=100, type_list=types2, force=True)

proc1.export(path="/home/azhangs/zed/crimepred_/pycode/api/TS1.p")
proc2.export(path="/home/azhangs/zed/crimepred_/pycode/api/TS2.p")
