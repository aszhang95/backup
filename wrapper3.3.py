import os, sys, numpy, scipy, sklearn
import os.path
from sklearn import manifold
from sklearn import cluster
import json, time, codecs
import subprocess as sp



class Z3_O:
	'''
    Stores key information about dataset including distance matrix, embedding
    of distance matrix and have the option to customize embedding algorithm
    parameters after defaults embeddings using the setter method with desired
    parameters

	attributes store the fit transformed feature (coordinate) matrix resulting from
    the specified embedding algorithm

	Inputs -
		dist_matrix (numpy n x n array): output of data smashing code
		embed_matrix_dict (dictionary): dictionary of feature matricies with key
            being embedding algorithm and value being the fit_transformed
            coordinate(feature) matrix
		feat_matrix_dict (dictionary): dictionary of information about the
        coordinate (feature)matrix with the key being the embedding algorithm and
            the value being the number of features produced by that algorithm
		clus_map (numpy n x x array): fit_transformed coordinates from KMeans
            clustering on distance matrix
	'''
	
	def __init__(self, dist_matrix, embed_matrix_dict, feat_matrix_dict, clus_map):
		self.distance_matrix = dist_matrix
		self.Sippl_embedding = embed_matrix_dict["sippl_emb"]
		self.MDS_embedding = embed_matrix_dict["mds_emb"]
		self.LLE_embedding = embed_matrix_dict["lle_emb"]
		self.Spectral_embedding = embed_matrix_dict["spec_emb"]
		self.Isomap_embedding = embed_matrix_dict["iso_emb"]
		self.Sippl_features = feat_matrix_dict["sippl"]
		self.MDS_features = feat_matrix_dict["mds"]
		self.LLE_features = feat_matrix_dict["lle"]
		self.Spectral_features = feat_matrix_dict["spec"]
		self.Isomap_features = feat_matrix_dict["iso"]
		self.num_Sippl_features = self.Sippl_features.shape[1]
		self.num_MDS_features = self.MDS_features.shape[1]
		self.num_LLE_features = self.LLE_features.shape[1]
		self.num_Spectral_features = self.Spectral_features.shape[1]
		self.num_Isomap_features = self.Isomap_features.shape[1]
		self.KMeans_cluster_fit = clus_map

	@property
	def num_Sippl_features(self):
		return self.num_Sippl_features

	@property
	def num_MDS_features(self):
		return self.num_MDS_features

	@property
	def num_LLE_features(self):
		return self.num_LLE_features

	@property
	def num_Spectral_features(self):
		return self.num_Spectral_features

	@property
	def num_Isomap_features(self):
		return self.num_Isomap_features

	@property
	def distance_matrix(self):
		return self.distance_matrix

	@property
	def Sippl_embedding(self):
		return self.Sippl_embedding

	@property
	def MDS_embedding(self):
		return self.MDS_embedding

	@property
	def LLE_embedding(self):
		return self.LLE_embedding

	@property
	def Spectral_embedding(self):
		return self.Spectral_embedding

	@property
	def Isomap_embedding(self):
		return self.Isomap_embedding

	@property
	def KMeans_cluster_fit(self):
		return self.KMeans_cluster_fit

	@property
	def Sippl_features(self):
		return self.Sippl_features

	@property
	def MDS_features(self):
		return self.MDS_features

	@property
	def LLE_features(self):
		return self.LLE_features

	@property
	def Spectral_features(self):
		return self.Spectral_features

	@property
	def Isomap_features(self):
		return self.Isomap_features

	@property
	def num_features(self):
		return self.num_features

	@MDS_embedding.setter
	def MDS_embedding(self, n_components=2, metric=True, n_init=4, max_iter=300,
		verbose=0, eps=0.001, n_jobs=1, random_state=None, dissimilarity='euclidean'):
		new_mds = manifold.MDS(n_components, metric, n_init, max_iter, verbose, eps, n_jobs,
			random_state, dissimilarity='euclidean')
		self.MDS_embedding = new_mds.fit_transform(self.distance_matrix)
		self.num_MDS_features = n_components

	@LLE_embedding.setter
	def LLE_embedding(self, n_neighbors=5, n_components=2, reg=0.001,
		eigen_solver='auto', tol=1e-06, max_iter=100, method='standard',
		hessian_tol=0.0001, modified_tol=1e-12, neighbors_algorithm='auto',
		random_state=None, n_jobs=1):
		new_lle = manifold.LocallyLinearEmbedding(n_neighbors, n_components, reg,
			eigen_solver, tol, max_iter, method, hessian_tol, modified_tol, neighbors_algorithm,
			random_state, n_jobs)
		self.LLE_embedding = new_lle.fit_transform(self.distance_matrix)
		self.num_LLE_features = n_components

	@Spectral_embedding.setter
	def Spectral_embedding(self, n_components=2, affinity='nearest_neighbors',
		gamma=None, random_state=None, eigen_solver=None, n_neighbors=None, n_jobs=1):
		new_spec = manifold.SpectralEmbedding(n_components, affinity, gamma, random_state, eigen_solver,
			n_neighbors, n_jobs)
		self.Spectral_embedding = new_spec.fit_transform(self.distance_matrix)
		self.num_Spectral_features = n_components

	@Isomap_embedding.setter
	def Isomap_embedding(self, n_neighbors=5, n_components=2, eigen_solver='auto',
		tol=0, max_iter=None, path_method='auto', neighbors_algorithm='auto', n_jobs=1):
		new_iso = manifold.Isomap(n_neighbors, n_components, eigen_solver, tol, max_iter, path_method,
		neighbors_algorithm, n_jobs)
		self.Isomap_embedding = new_iso.fit_transform(self.distance_matrix)
		self.num_Isomap_features = n_components

	@KMeans_cluster_fit.setter
	def KMeans_cluster_fit(self, n_clusters=8, init='k-means++', n_init=10, max_iter=300, tol=0.0001,
		precompute_distances='auto', verbose=0, random_state=None,
		copy_x=True, n_jobs=1, algorithm='auto'):
		new_kmeans = cluster.KMeans(n_clusters, init, n_init, max_iter, tol,
		precompute_distances, verbose, random_state, copy_x,
		n_jobs, algorithm)
		self.KMeans_cluster_fit = new_kmeans.fit_transform(self.distance_matrix)

	def write_out_json(self, outfile_name="Z3_O_stats.json"):
		'''
		Writes matrices of Z3_obj out as a json file titled "Z3_O_stats.json" if no 
		name specified

		Inputs -
			outfile_name (string): desired file name of exported data as json

		Outputs -
			json of Z3_O data [default name is "Z3_O_stats.json"]
		'''

		rv = { "distance_matrix":self.distance_matrix.tolist(),
		"Spectral_embedding":{"features":self.Spectral_features.tolist(),
		"embedding":self.Spectral_embedding.tolist()},
		"Sippl_embedding":{"features":self.Sippl_features.tolist(),
		"embedding":self.Sippl_embedding.tolist()},
		"MDS_embedding":{"features":self.MDS_features.tolist(),
		"embedding":self.MDS_embedding.tolist()},
		"LLE_embedding":{"features":self.LLE_features.tolist(),
		"embedding":self.LLE_embedding.tolist()},
		"Isomap_embedding":{"features":self.Isomap_features.tolist(),
		"embedding":self.Isomap_embedding.tolist()},
		"Kmeans_clustering_matrix":self.KMeans_cluster_fit.tolist() }

		with open(outfile_name, 'w') as outfile:
			json.dump(rv, outfile)



def Z3_master(Input_object_corpora, time_data_loc, input_type="time_series", is_symbolic=True,
	with_class_labels=False, partition=None, get_embedding=True,
    get_classification=False, classification_algorithm="KNN",
    Classification_algorithm_parameters=None, generate_features=False, num_features=2):
	'''
	Ishanu's description:
    Master primitive encompassing Z3, Z3-C, Z3-E, Z3-F
    as well as the packaged versions with ZS-ZQ- preprocessing
    Input_object_corpora can be of type time_series
    (continuous valued, or Tn valued), or
    can be of type X (See definition in Table 2), when ZS, ZQ will be called
    internally

	Calls embedding algorithm on the distance_matrix and creates Z3_object to store
	this information

	Inputs -
		Input object corpora (np n x n array): output of data smashing
		time_data_loc (relative directory path as string): path to input of data smashing code


	Outputs -
		Z3_obj (object): Z3 object containing information about input dataset
    '''

	sp.Popen('./bin/embed -f ' + time_data_loc, shell=True).wait()
	sippl_embed = numpy.loadtxt(fname="outE.txt", dtype=float)
	sippl_feat = sippl_embed[:, :num_features]

	n_components, n_neighbors = num_features, 10

	se = manifold.SpectralEmbedding(n_components=n_components, n_neighbors=n_neighbors)
	se_fit = se.fit_transform(Input_object_corpora)

	mds = manifold.MDS(n_components, max_iter=100, n_init=1)
	mds_fit = mds.fit_transform(Input_object_corpora)

	lle = manifold.LocallyLinearEmbedding(n_components=n_components, n_neighbors=n_neighbors)
	lle_fit = lle.fit_transform(Input_object_corpora)

	iso = manifold.Isomap(n_components=n_components, n_neighbors=n_neighbors)
	iso_fit = iso.fit_transform(Input_object_corpora)

	embed_dict = {"sippl_emb":sippl_embed, "spec_emb":se_fit, "mds_emb":mds_fit,
	"lle_emb":lle_fit, "iso_emb":iso_fit}

	features_dict = {"sippl":sippl_feat, "spec":se_fit,"mds":mds_fit,
	"lle":lle_fit, "iso":iso_fit}

	kmeans_fit = cluster.KMeans().fit_transform(Input_object_corpora)
	result_obj = Z3_O(dist_matrix=Input_object_corpora, embed_matrix_dict=embed_dict,
    feat_matrix_dict=features_dict, clus_map=kmeans_fit)

	return result_obj



def load_Z3_obj(json_name="Z3_O_stats.json"):
	'''
	Reads a json of Z3_O data into a Z3_O object or None if the json specified cannot be found

	Inputs -
		json_name (string): the name of the json containing Z3_O data

	Returns -
		(Z3_O object)
	'''

	if "pycode" in os.getcwd():
		os.chdir("..")

	if os.path.isfile(json_name):
		with open(json_name) as data_file:
			data = json.load(data_file)
			dist_mat = numpy.array(data["distance_matrix"])
			embed_names = ["Spectral_embedding", "Sippl_embedding", "MDS_embedding",
			"LLE_embedding", "Isomap_embedding"]
			embed_dict, feat_matrix = {}, {}
			for name in embed_names:
				if 'spec' in name.lower():
					embed_dict['spec_emb'] = numpy.array(data[name]["embedding"])
					feat_matrix['spec'] = numpy.array(data[name]["features"])
				elif 'sippl' in name.lower():
					embed_dict['sippl_emb'] = numpy.array(data[name]["embedding"])
					feat_matrix['sippl'] = numpy.array(data[name]["features"])
				elif 'mds' in name.lower():
					embed_dict['mds_emb'] = numpy.array(data[name]["embedding"])
					feat_matrix['mds'] = numpy.array(data[name]["features"])
				elif 'lle' in name.lower():
					embed_dict['lle_emb'] = numpy.array(data[name]["embedding"])
					feat_matrix['lle'] = numpy.array(data[name]["features"])
				elif 'iso' in name.lower():
					embed_dict['iso_emb'] = numpy.array(data[name]["embedding"])
					feat_matrix['iso'] = numpy.array(data[name]["features"])
			kmeans = numpy.array(data['Kmeans_clustering_matrix'])

			return Z3_O(dist_matrix=dist_mat, embed_matrix_dict=embed_dict,
			feat_matrix_dict=feat_matrix, clus_map=kmeans)

	else:
		print("Error: could not find Z3_stats json in current directory.")
		return None



if __name__ == '__main__':
	sp.Popen("module load python", shell=True).wait()
	print("Running smash.cc algorithm...")
	print("Loading modules...")

	sp.Popen("module load gsl", shell=True).wait()
	sp.Popen("module load boost/1.63.0+gcc-6.2", shell=True).wait()
	
	os.chdir("..")
	pwd = os.curdir
	if not os.path.isdir(pwd + "/bin"):
		os.chdir("zbase")
		sp.Popen("make -f Makefile", shell=True).wait()
		os.chdir("..")
		sp.Popen("make -f Makefile", shell=True).wait()

	print("Dependencies satisfied.")

	input_str = ''
	symbolic, results_fname = False, "H.dst"
	tdata_loc = ''
	for i in range(len(sys.argv)):
		if i == 0:
			continue
		if sys.argv[i-1] == '-o':
			results_fname = sys.argv[i]
		elif sys.argv[i-1] == '-f':
			tdata_loc = sys.argv[i]
		if sys.argv[i] == 'symbolic':
			symbolic = True
		input_str += sys.argv[i] + ' '

	print("Requested: " + input_str)
	sp.Popen('./bin/smash ' + input_str, shell=True).wait()

	while not os.path.isfile(results_fname):
		print("Waiting for smash algorithm to complete...")
		time.sleep(20)

	data = numpy.loadtxt(fname=results_fname, dtype=float)

	raw_input("Enter the number of features to generate for each of the default embeddings. \
		These can be changed later by calling the embedding method again once the object \
		has been created.")
	out = Z3_master(Input_object_corpora=data, time_data_loc=tdata_loc,
                    is_symbolic=symbolic)

	out.write_out_json()
	print("\n\n\n Complete.")
