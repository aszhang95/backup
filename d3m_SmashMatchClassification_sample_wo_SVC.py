# start from working directory of where cw8 is located
from d3m_SmashMatchClassification import *

# Classification of TEST0 from ../data_small using Smashmatch
bin_path = "../bin/smashmatch"

# define quantizer function
def q(x):
    if x <= 0:
        return 0
    else:
        return 1

# create instance of the SmashMatchClassification with the given quantization and force vectorization
# note: force_vect_preproc == True by default, unless input quantizer is vectorized,
# then need to set as False
clf = SmashMatchClassification(bin_path, preproc_=q, force_vect_preproc=True)

# quantize and read in library files as pd.DataFrame
lib0 = clf.read_series("../data_small/LIB0", delimiter_=" ", quantize=True)
lib1 = clf.read_series("../data_small/LIB1", delimiter_=" ", quantize=True)
lib2 = clf.read_series("../data_small/LIB2", delimiter_=" ", quantize=True)

# map library files to class/label numbers
# i.e. all timeseries in lib0 are of class 0, etc.
maps = [(lib0, 0), (lib1, 1), (lib2, 2)]

# create input data from previous mappings to be fit into SmashMatchClassification
X, y = clf.condense(maps)

# fit the formatted examples and labels
# since when reading in the libraries we had alreaady performed quantization,
# we do not need to quantize again
clf.fit(X, y)

# quantize and read in the timeseries data to be classified as pd.DataFrame
data = clf.read_series("../data_small/TEST0", delimiter_=" ", quantize=True)

# run predict twice, and print the results
print(clf.predict(data, nr=2))

# get the log probabilities for each timeseries to fall in each class,
# and force rerun of the SmashMatch algorithm
print(clf.predict_log_proba(data, nr=2, force=True))
