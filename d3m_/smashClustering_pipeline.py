#!/usr/bin/python

from smashClustering import *
from primitives_interfaces.utils.series import *
from sklearn import cluster

# declare bin location relative to script path
bin_path = "./data_smashing_/bin"

# Reading in data from deploy_scripts/examples/data.dat and setting dtype=np.int32
# because this input data is categorical
X = read_series("./data_/data.dat", delimiter=" ").values.astype(np.int32)

# define quantizer function
def q(x):
    if x <= 0:
        return 0
    else:
        return 1

# create Input class instance
data_class = Input(data=X, is_categorical=True, is_synchronized=True, preproc=q)

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
KMeans_cc = cluster.KMeans(n_clusters=4)
scc.cluster_class = KMeans_cc

# can also call fit_predict for convenience (note: distance matrix will be recalculated)
print(scc.fit_predict(nr=3))

# second test on known dataset of 3 classes
X2 = read_series("./data_/COMBINED_TEST1", delimiter=" ").values
data_class2 = Input(data=X2, is_categorical=True, is_synchronized=True,\
preproc=q)
scc.data = data_class2

# create the clustering classes to use (meanshift does not take param n_clusters, so reuse)
KMeans_cc2 = KMeans_cc = cluster.KMeans(n_clusters=3)

# compare
scc.cluster_class = KMeans_cc2
print(scc.fit(nr=3))
print(scc.fit_predict(nr=3))
scc.cluster_class = meanshift_cc
print(scc.fit(nr=3))
print(scc.fit_predict(nr=3))


# can switch clustering class; first need to initialize then set
# is_affinity=True implies we are usig an affinity matrix, so larger values
# imply more similarity, as opposed to a distance matrix
Spectral_cc = cluster.SpectralClustering(n_clusters=3,affinity='precomputed')
scc.cluster_class = Spectral_cc
print "spectral clustering: ", scc.fit_predict(nr=3,is_affinity=True)
