### Sample D3M Classification Wrapper usage

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
lib0 = test.read_in_ragged("LIB0", " ")
lib1 = test.read_in_ragged("LIB1", " ")
lib2 = test.read_in_ragged("LIB2", " ")

# to format into sklearn.SVM X, y inputs, need to run a few helper commands
class_of_lib0 = 0
class_of_lib1 = 1
class_of_lib2 = 2

mappings = [(lib0, class_of_lib0), (lib1, class_of_lib1), (lib2, class_of_lib2)]

# then to create X, y inputs
X, y = test.condense(mappings)

# now ready to fit
test.fit(X, y)

#can check if labels for each class are confirgured correctly
test.classes

# then to read in a test data file for classification,
data = test.read_in_ragged("../../cwtest/TEST1", delimiter_=" ", datatype=int)

# can run test.predict, test.predict_proba (which returns the percentage probabilities),
# and test.predict_log_proba
results = test.predict(data)


'''
Known Issue:
    bin/smashmatch segfaults if given paths to files as opposed to simply filenames?
    other possible issue is library file timeseries must be ints?
    the above sample code was run to produce the error seen in cw4_segfault.png
'''
