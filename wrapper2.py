import os, subprocess, sys, numpy, scipy, sklearn
import os.path
from sklearn import manifold
from sklearn import cluster
import json, time 

''' The below are required for sklearn classification '''
# from sklearn.cross_validation import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.svm import SVC
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.lda import LDA
# from sklearn.qda import QDA




'''
Progress (8/7/2017): 
    - actual coordinates (features) = outcome of embedding function, stored as embed_matrix, run on 
    distance matrix; custom params for embedding function can be set using setter method of Z3_object
    - outlined classifier code to run all sklearn classifiers on dataset, if desired; 
    need lables/targets to be able to run
    - KMeans_clustering called, not necessary, but kept in case useful. Can be deleted
'''

class Z3_O:
	# def __init__(self, dist_matrix, embed_matrix_dict, feat_matrix, clus_map,
	# 	class_probability, s_ann_error, conf_bnd_L, conf_bnd_U):
	# 	self.distance_matrix = dist_matrix
	
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
		self.Sippl_embedding = embedding_matrix_dict["sippl_emb"]
		self.MDS_embedding = embedding_matrix_dict["mds_emb"]
		self.LLE_embedding = embedding_matrix_dict["lle_emb"]
		self.Spectral_embedding = embedding_matrix_dict["spec_emb"]
		self.Isomap_embedding = embedding_matrix["iso_emb"]
		self.feature_matrix_dict = feat_matrix_dict
		self.KMeans_cluster_fit = clus_map
            
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
	def Sippl_num_features(self):
		return self.feat_matrix_dict['sippl']

	@property
	def MDS_num_features(self):
		return self.feat_matrix_dict['mds']

	@property
	def LLE_num_features(self):
		return self.feat_matrix_dict['lle']

	@property
	def Spectral_num_features(self):
		return self.feat_matrix_dict['spec']

	@property 
	def Isomap_num_features(self):
		return self.feat_matrix_dict['iso']
    
    # if want to use custom embedding methods/classifying methods, 
    # would need setter methods in the object itself
    # for now, comes with these defaults
    
	@MDS_embedding.setter
	def MDS_embedding(self, n_components=2, metric=True, n_init=4, max_iter=300, 
		verbose=0, eps=0.001, n_jobs=1, random_state=None, dissimilarity='euclidean'):
		new_mds = manifold.MDS(n_components, metric, n_init, max_iter, verbose, eps, n_jobs,
			random_state, dissimilarity='euclidean')
		self.MDS_embedding = new_mds.fit_transform(self.distance_matrix)
		self.feature_matrix_dict['mds'] = n_components
    
	@LLE_embedding.setter 
	def LLE_embedding(self, n_neighbors=5, n_components=2, reg=0.001,
		eigen_solver='auto', tol=1e-06, max_iter=100, method='standard', 
		hessian_tol=0.0001, modified_tol=1e-12, neighbors_algorithm='auto', 
		random_state=None, n_jobs=1):
		new_lle = manifold.LocallyLinearEmbedding(n_neighbors, n_components, reg, 
			eigen_solver, tol, max_iter, method, hessian_tol, modified_tol, neighbors_algorithm, 
			random_state, n_jobs)
		self.LLE_embedding = new_lle.fit_transform(self.distance_matrix)
		self.feature_matrix_dict['lle'] = n_components
        
	@Spectral_embedding.setter
	def Spectral_embedding(self, n_components=2, affinity='nearest_neighbors',
		gamma=None, random_state=None, eigen_solver=None, n_neighbors=None, n_jobs=1):
		new_spec = manifold.SpectralEmbedding(n_components, affinity, gamma, random_state, eigen_solver,
			n_neighbors, n_jobs)
		self.Spectral_embedding = new_spec.fit_transform(self.distance_matrix)
		self.feature_matrix_dict['spec'] = n_components
    
	@Isomap_embedding.setter
	def Isomap_embedding(self, n_neighbors=5, n_components=2, eigen_solver='auto',
		tol=0, max_iter=None, path_method='auto', neighbors_algorithm='auto', n_jobs=1):
		new_iso = manifold.Isomap(n_neighbors, n_components, eigen_solver, tol, max_iter, path_method, 
		neighbors_algorithm, n_jobs)
		self.Isomap_embedding = new_iso.fit_transform(self.distance_matrix)
		self.feature_matrix_dict['iso'] = n_components

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
		Writes matrices of Z3_obj out as a json file titled "Z3_O_stats.json"

		Inputs - 
			outfile_name (string): desired file name of exported data as json
		'''
		with open('outfile_name', 'w') as outfile:
			rv = { "distance_matrix":self.dist_matrix,
			"Spectral_embedding":{"num_features":self.feature_matrix_dict['spec'],
			"feat_matrix":self.Spectral_embedding},
			"Sippl_embedding":{"num_features":self.feature_matrix_dict['sippl'], 
			"feat_matrix":self.Sippl_embedding},
			"MDS_embedding":{"num_features":self.feature_matrix_dict['mds'], 
			"feat_matrix":self.MDS_embedding},
			"LLE_embedding":{"num_features":self.feature_matrix_dict['lle'], 
			"feat_matrix":self.LLE_embedding},
			"Isomap_embedding":{"num_features":self.feature_matrix_dict['iso'],
			"feat_matrix":self.Isomap_embedding}, 
			"Kmeans_clustering_matrix":self.KMeans_cluster_fit }
			json.dump(rv, outfile)



#	to be implemented:
# 		self.class_probab = class_probability
# 		self.self_ann_error = s_ann_error
# 		self.confidence_bnd_L = conf_bnd_L
# 		self.confidence_bnd_U = conf_bnd_U

		

def Z3_master(Input_object_corpora, time_data_loc, input_type="time_series", is_symbolic=True,
	with_class_labels=False, partition=None, get_embedding=True,
    # embedding_algorithm="Sippl",
    # embedding_algorithm_parameters=None,
    get_classification=False, classification_algorithm="KNN", 
    Classification_algorithm_parameters=None, generate_features=False):

	'''
	Ishanu's description:
    Master primitive encompassing Z3, Z3-C, Z3-E, Z3-F
    as well as the packaged versions with ZS-ZQ- preprocessing
    Input_object_corpora can be of type time_series
    (continuous valued, or Tn valued), or
    can be of type X (See definition in Table 2), when ZS, ZQ will be called 
    internally
    
	Calls embedding algorithm on the distance_matrix and creates Z3_object to store
	the information

	Inputs - 
		Input object corpora (np n x n array): output of data smashing 
		time_data_loc (relative directory path as string): path to input of data smashing code


	Outputs - 
		Z3_obj (object): Z3 object containing information about input dataset
    '''

    # embedding algorithms: Sippl
	subprocess.Popen('./bin/embed -f ' + time_data_loc, shell=True)
	sippl_embed = numpy.loadtxt(fname="outE.txt", dtype=float)

	# embedding algorithms via sklearn default parameters
	n_components, n_neighbors = 2, 10

	# now call subprocess to load python module so as not to conflict 
	# with other processes
	subprocess.Popen("module load python", shell=True)

	# embedding algorithms: Spectral
	se = manifold.SpectralEmbedding(n_components=n_components,
                                n_neighbors=n_neighbors)
	se_fit = se.fit_transform(Input_object_corpora)

	# embedding algorithms: MDS
	mds = manifold.MDS(n_components, max_iter=100, n_init=1)
	mds_fit = mds.fit_transform(Input_object_corpora)

    #embedding algorithms: Locally Linear Embedding
	lle = manifold.LocallyLinearEmbedding(n_components=n_components, n_neighbors=n_neighbors)
	lle_fit = lle.fit_transform(Input_object_corpora)
    
    #embedding algorithms: Isomap
	iso = manifold.Isomap(n_components=n_components, n_neighbors=n_neighbors)
	iso_fit = iso.fit_transform(Input_object_corpora)
    
	embed_dict = {"sippl_emb":sippl_embed, "spec_emb":se_fit, "mds_emb":mds_fit, 
	"lle_emb":lle_fit, "iso_emb":iso_fit}
    
    # features_dict only stores num_dimensions for features for now until further clarification
	features_dict = {"sippl":embed_dict['sippl_emb'].shape[0], "spec":n_components,"mds":n_components,
	"lle":n_components}
	
	'''
    The below are supervised clusterings
    
    names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Decision Tree",
             "Random Forest", "AdaBoost", "Naive Bayes", "LDA", "QDA"]
    classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025),
        SVC(gamma=2, C=1),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        AdaBoostClassifier(),
        GaussianNB(),
        LDA(),
        QDA()]
    
    # specific to this case of 20 datastreams, need clarification on target
    # which is the categories to fit the clusters into 
    # supervised??
    y = np.array(range(1,21))
    
    # preprocess dataset, split into training and test part
    X = StandardScaler().fit_transform(X)
    X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=.4)
    
    classification_dict = {}
    # iterate over classifiers
    # y_train is the labels - labels are just embed matrix?
    for name, clf in zip(names, classifiers):
        clf.fit(X_train, y_train)
        classification_dict[name] = clf.score(X_test, y_train)
    '''
    
    # unsupervised, not necessary?
	kmeans_fit = cluster.KMeans().fit_transform(Input_object_corpora)
    
        
	result_obj = Z3_O(dist_matrix=Input_object_corpora, embed_matrix_dict=embed_dict, 
		feat_matrix_dict=features_dict, clus_map=kmeans_fit)
    
	return result_obj
	
	
if __name__ == '__main__':
	print("Running smash.cc algorithm...")

	# assumes is running from pycode directory
	# check to see if bin directory exists, otherwise, need to setup
	os.chdir("..")
	pwd = os.curdir
	if not os.path.isdir(pwd + "/bin"):
		os.chdir("zbase")
		# subprocess.Popen("module load gsl", shell=True)
		# subprocess.Popen("module load boost/1.63.0+gcc-6.2", shell=True)
        '''
        current issue: 
        /bin/sh: ./bin/smash: No such file or directory
        WARNING: openmpi/2.0.2+gcc-6.2 cannot be loaded due to a conflict.
        HINT: Might try "module unload openmpi" first.
        WARNING: python/2.7.13+gcc-6.2 cannot be loaded due to a conflict.
        HINT: Might try "module unload python" first.	
        '''
		subprocess.Popen("make config.o", shell=True)
		subprocess.Popen("make semantic.o", shell=True)
		os.chdir("..")
		subprocess.Popen("make -f Makefile_no_static", shell=True)
		print("Loading modules...")

	input_str = ''
	symbolic, results_fname = False, "H.dst"
	tdata_loc = ''
	for i in range(len(sys.argv[1:])):
		if sys.argv[i-1] == '-o':
			results_fname = sys.argv[i]
		elif sys.argv[i-1] == '-f':
			tdata_loc = sys.argv[i]
		if sys.argv[i] == 'symbolic':
			symbolic = True
		input_str += sys.argv[i] + ' '

	print("Dependencies satisfied.")
	print("Requested: " + input_str)
	subprocess.Popen('./bin/smash ' + input_str, shell=True)

	# create numpy matrix of output data
	# assume numpy matrix by row
	# have to wait for the actual smashing code to work and produce distance matrix
	while not os.path.isfile(results_fname):
		print("Waiting for smash algorithm to complete...")
		time.sleep(20)

	data = numpy.loadtxt(fname=results_fname, dtype=float)
	out = Z3_master(Input_object_corpora=data, time_data_loc=tdata_loc,
                    is_symbolic=symbolic)

	out.write_out_json()