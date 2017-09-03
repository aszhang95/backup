### Sample D3M Classification Wrapper usage

# from within ipython:
from cw6 import *
# parameter of SupervisedModelBase: path/to/bin/smashmatch
bin_path = os.path.abspath("../bin/smashmatch")

# then initiate SupervisedModelBase class
test = SupervisedModelBase(bin_path)

# if data not already formatted in sklearn.SVM classifier input format, can read in time series from file
# read_in_ragged parameters: path/to/file, delimiter
# optional third parameter: datatype
# however for reading in library files of timeseries of unequal length
# leave third parameter as default otherwise pandas will try to format NaN's as ints which will
# throw an exception
lib0 = test.read_in_ragged("data_small/LIB0", " ")
lib1 = test.read_in_ragged("data_small/LIB1", " ")
lib2 = test.read_in_ragged("data_small/LIB2", " ")

# to format into sklearn.SVM X, y inputs, need to run a few helper commands
class_of_lib0 = 0
class_of_lib1 = 1
class_of_lib2 = 2

mappings = [(lib0, class_of_lib0), (lib1, class_of_lib1), (lib2, class_of_lib2)]

# or, if dataset is vertically oriented, and each row is a value
# and multiple files comprise a clase - consolidate those files into one folder
# and run (note this returns a tuple)
# mapping0 = test.read_in_vert("CHE0-30", 1, 0, 0, 1, float)
# mapping1 = test.read_in_vert("JOH0-30", 2, 0, 0, 1, float)
# mappings = [mapping0, mapping1]

# then to create X, y inputs
X, y = test.condense(mappings)

# now ready to fit
test.fit(X, y)

#can check if labels for each class are confirgured correctly
test.classes

# then to read in a test data file for classification,
data = test.read_in_ragged("data_small/TEST0", delimiter_=" ", datatype=int)

# can run test.predict, test.predict_proba (which returns the percentage probabilities),
# and test.predict_log_proba
results = test.predict(data)
print results
