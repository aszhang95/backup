# import necessary libraries and utility files
from smashDistanceMetricLearning import *
from seriesUtil import *

# declare bin location relative to script path
bin_path = "../bin"

# Reading in data from deploy_scripts/examples/data.dat and setting dtype=np.int32
# because this input data is categorical
X = read_series("../deploy_scripts/examples/data.dat", delimiter=" ").values.astype(np.int32)

# define quantizer function
def q(x):
    if x <= 0:
        return 0
    else:
        return 1

# create Input class instance and vectorize quantizer (note: vectorizing
# quantizer is default behavior but can be set to False if quantizer already vectorized)
data_class = Input(data=X, is_categorical=True, is_synchronized=True,\
preproc=q, force_vect_preproc=True)


# create SmashDistanceMatrixLearning class to run methods
dmlc = SmashDistanceMetricLearning(bin_path=bin_path, input_class=data_class)

# return distance matrix of input timeseries data (repeat calculation 3 times)
print(dmlc.fit(nr=3))
