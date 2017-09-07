from d3m_SmashClustering import *
from d3m_series_util import *
from sklearn import cluster

# declare bin location relative to script path
bin_path = "../bin"

# Reading in data from deploy_scripts/examples/data.dat and setting dtype=np.int32
# because this input data is categorical
X = read_in_series("../deploy_scripts/examples/data.dat", delimiter=" ").values.astype(np.int32)

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


# decide now many clusters to predict
num_clusters = 4

# instantiate clustering class to be used to cluster distance matrix data
# if not specified, default is cluster.KMeans
kmeans_cc = cluster.KMeans(n_clusters=num_clusters)

# create SmashClustering class to run methods (require 4 clusters)
# create sklearn.cluster.KMeans class with defaults since instance given
scc = SmashClustering(bin_path_=bin_path, input_class_=data_class, n_clus=num_clusters)

# return distance matrix of input timeseries data (repeat calculation 3 times)
print(scc.fit(nr=3))

# standard sklearn.cluster.KMeans.fit_predict operation on the distance matrix
print(scc.fit_predict(nr=3))

# standard sklearn.cluster.KMeans.fit_transform operation on the distance matrix
print(scc.fit_transform(nr=3))

# standard sklearn.cluster.KMeans.predict operation on the distance matrix
print(scc.predict(nr=2))

#standard sklearn.cluster.KMeans.score operation on the distance matrix
print(scc.score(nr=2))

# standard sklearn.cluster.KMeans.transform operation on the distance matrix
print(scc.transform(nr=2))
