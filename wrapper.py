
def Z3_master_(Input_object_corpora,
               input_type="time_series",
               is_symbolic=True,
               with_class_labels=False, 
               partition=None,
               get_embedding=True,
               embedding_algorithm="Sippl",
               embedding_algorithm_parameters=None,
               get_classification=False,
               classification_algorithm="KNN",
               Classification_algorithm_parameters=None,
               generate_features=False):


    # Master primitive encompassing Z3, Z3-C, Z3-E, Z3-F
    # as well as the packaged versions with ZS-ZQ- preprocessing
    # Input_object_corpora can be of type time_series
    # (continuous valued, or Tn valued), or
    # can be of type X (See definition in Table 2), when ZS, ZQ will be called internally


    return z3o



 
Class Z3-O(object):
    """ return object from Z3_master_.
    Attributes:
        distance_matrix:    numpy.matrix of shape (N,N) with float entries 
                            where int N is the number of samples
        embedding_matrix:   numpy.matrix of shape (N,d) with float entries 
                            where int N is the number of samples, int d is the 
                            embedding dimension (default: None)
        feature_matrix:     numpy.matrix of shape (N,d) with float entries 
                            where int N is the number of samples, int d is the 
                            feature dimension (default: None)
        cluster_map:        numpy.array with int entries of shape (N) (default: None)
        class_prob:         numpy.matrix with float entries of shape (N,d) 
                            where int d is the number of training classes
                            Here roes add to 1. (default: None)
        self_ann_error:     numpy.array of shape (N) with float entries
        Confidence_bnd_L:   lower confidence bound on distance_matrix.
                            numpy.array with int entries of shape (N) 
        Confidence_bnd_U:   upper confidence bound on distance_matrix.
                            numpy.array with int entries of shape (N) 
    """
