# import relevant libraries
from smashFeaturization import *
from seriesUtil import *
from sklearn import manifold

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
data_class = Input(data=X, is_categorical=True, is_synchronized=True, preproc=q)

# decide on number of features to use (default is 2)
num_f = 2

# instantiate another featurizing class if desired
# e.g. sklearn.manifold.MDS
mds_feat = manifold.MDS(n_components=num_f, dissimilarity="precomputed")

# create SmashEmbedding class to run methods (require 2 dimensions in embedding)
sfc = SmashFeaturization(bin_path=bin_path, \
input_class=data_class, n_feats=num_f, feat_class=mds_feat)

# return distance matrix of input timeseries data (repeat calculation 3 times)
# NOTE: fits both default Sippl Embedding and user-defined custom embedding class
print(sfc.fit(nr=3))

# return embedded coordinates using Sippl embedding (default) on distance matrix
print(sfc.fit_transform(nr=3, featurizer='default'))

# return embedded coordinates using Sippl embedding (default) on distance matrix
print(sfc.fit_transform(nr=3, featurizer='custom'))
