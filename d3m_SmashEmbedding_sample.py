# import relevant libraries
from d3m_SmashEmbedding import *
from d3m_series_util import *
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
data_class = Input(data=X, is_categorical=True, is_synchronized=True,\
preproc=q, force_vect_preproc=True)

# create SmashEmbedding class to run methods (require 2 dimensions in embedding)
sec = SmashEmbedding(bin_path_=bin_path, input_class_=data_class, n_dim=2)

# return distance matrix of input timeseries data (repeat calculation 3 times)
print(sec.fit(nr=3))

# return embedded coordinates using Sippl embedding (default) on distance matrix
print(sec.fit_transform(nr=3))

# use other embedding function to embed the data e.g. sklearn.manifold.MDS
mds_emb = manifold.MDS(n_components=sec.num_dimensions, dissimilarity="precomputed")
print(sec.fit_transform(nr=3, embedder=mds_emb))
