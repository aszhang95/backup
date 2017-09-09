from smashClustering import *
from seriesUtil import *
from sklearn import cluster

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


# instantiate clustering class to be used to cluster distance matrix data
# if not specified, default is cluster.KMeans
meanshift_cc = cluster.MeanShift()

# instantiate clustering class
# default cluster_class is sklearn.cluster.KMeans and numCluster default is 8
scc = SmashClustering(bin_path=bin_path, input_class=data_class, cluster_class=meanshift_cc)

# fit and return distance matrix of input timeseries data (repeat calculation 3 times)
print(scc.fit(nr=3))

# call clustering class predict method on fitted input data
print(scc.predict(nr=3))

# can switch clustering class; first need to initialize then set
KMeans_cc= cluster.KMeans(n_clusters=4)
scc.cluster_class = KMeans_cc

# can also call fit_predict for convenience (note: distance matrix will be recalculated)
print(scc.fit_predict(nr=3))
