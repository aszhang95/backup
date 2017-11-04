# Sample script for crimedataproc2.py useage

from crimedataproc2 import *

# Instantiate the class; path & bin_path are required, but for time, large dataset already processed
# so can load from the temporary directory called f248b3715732426285d1f439e3e27b23 within pycode/api
proc = CrimeData(path="../../data/Crimes_-_2001_to_present.csv", bin_path="./bin/procdata", \
file_dir="f248b3715732426285d1f439e3e27b23")

# large dataset already processed, meta dictionary stored as a pickle dictionary called meta_dict.p
 # and can be loaded for use
meta = pickle.load(open("f248b3715732426285d1f439e3e27b23/meta_dict.p", "rb"))

# then, to run transform to test bin/preproc (transform is the method that actually calls bin/preproc)
# load the meta_dict that with the necessary attributes (min/max dates, min/max lon/lat pairs)
# into the proc class
# Note: if running for the first time, can just make straight call to transform

proc.meta_dict = meta

# then call transform; set the grid size and can set force=True to force preproc before calling
# bin/procdata
loc_ts_df = proc.transform(grid_size=200, force=False)

# Necessary bugfix:
# try calling proc.indexmap - will return an emtpy dataframe, and cd-ing into the temporary folder
# where all the temporary files/pickle files are stored, to read "200" (or whatever you set the
# grid size to be) shows that no longitude/latitude pairs show up in the location-index file
# just zeroes (see also the bug screenshot)
proc.indexmap

# Other note:
# to fix the hanging I now use the less precise lat/lon pairs - I can change this back if
# more precision is required, but I just wanted to have something to give you by Wednesday

# also single useful meta information that I made into an attribute for convenience
proc.num_crimesites
