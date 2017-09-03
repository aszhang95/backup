import os, csv, pdb, tempfile, ntpath, uuid, atexit, sys
import subprocess as sp
import numpy as np
from numpy import nan
import pandas as pd
from sklearn import cluster



# Potential known bug: does tempfile create NamedTemporaryFile names with hyphens?



# global variables
global cwd
cwd = os.getcwd()
global temp_dir
temp_dir = str(uuid.uuid4())
temp_dir = temp_dir.replace("-", "")


# global helper functions
def path_leaf(path):
    '''
    Returns filename from a given path/to/file
    Taken entirely from Lauritz V. Thaulow on https://stackoverflow.com/questions/8384737

    Input -
        path (string): path/to/the/file

    Returns -
        filename (string)
    '''

    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)



class SmashClustering:
    '''
    Object for running Smashmatch to calculate the distance matrix between time series
    and using sklearn.cluster.KMeans to perform unsupervised clustering

    Inputs -
        bin_path_(string): Path to smashmatch as a string
        quantiziation (function): quantization function for time series data

    Attributes:
        bin_path (string): path to bin/smash
        quantizer (function): function to quantify input data
        quantized input (numpy.ndarray or pd.DataFrame) the transformed input data if
            quantizer function is specified
        num_clusters (int): number of clusters used for embedding

        (Note: bin_path and num_clusters can be set by assignment, input and quantizer must be
            set using custom method)
    '''

    def __init__(self, bin_path_, X, n_clus, quantizer_=None, KMeans_class_=None):
        self.bin_path = bin_path_
        self.__quantizer = quantizer_
        self.__input_dm = X
        self.__num_clusters = n_clus
        if KMeans_class_ is not None:
            self.__KMeans_class = KMeans_class_
        else:
            self.__KMeans_class = cluster.Kmeans(n_clusters=self.__num_clusters)
        prev_wd = os.getcwd()
        os.chdir(cwd)
        sp.Popen("mkdir "+ temp_dir, shell=True).wait()
        self.__file_dir = cwd + "/" + temp_dir
        os.chdir(prev_wd)
        if quantizer is not None:
            if isinstance(self.__input_dm, pd.DataFrame):
                self.__quantized_input = self.__input_dm.applymap(self.__quantizer)
            elif isinstance(self.__input_dm, np.ndarray):
                self.quantizer = np.vectorize(self.quantizer)
                self.__quantized_input = self.quantizer(self.__input_dm)
            else:
                raise ("Error: input data must be either numpy.ndarray or pandas.DataFrame.")
        else:
            self.__quantized_input = None
        self.__input_dm_fh = None
        self.__input_dm_fname = None
        self.__output_dm_fname = None
        self.__command = (self.bin_path + "/smash")
        self.__input_e = None

    @property
    def input(self):
        return self.__input


    @property
    def quantizer(self):
        return self.__quantizer


    @property
    def quantized_input(self):
        return self.__quantized_input


    @property
    def bin_path(self):
        return self.__bin_path


    @property
    def num_clusters(self):
        return self.__num_clusters


    @property
    def KMeans_class(self):
        return self.__KMeans_class


    @bin_path.setter
    def bin_path(self, new_bin_path):
        self.__bin_path = new_bin_path
        self.__command = self.__bin_path


    @num_clusters.setter
    def num_clusters(self, new_nclus):
        self.__num_clusters = new_nclus


    @KMeans_class.setter
    def KMeans_class(self, new_kmeans_class):
        self.__KMeans_class = new_kmeans_class


    def set_input(self, new_input, new_quantizer=None):
        '''
        Setter method for class input attributes; modifies object in place
        No I/O
        '''

        self.__input_dm = new_input
        if new_quantizer is not None:
            self.__quantizer = new_quantizer
            if isinstance(self.__input_dm, pd.DataFrame):
                self.__quantized_input = self.__input_dm.applymap(self.__quantizer)
            elif isinstance(self.__input_dm, np.ndarray):
                self.quantizer = np.vectorize(self.quantizer)
                self.__quantized_input = self.quantizer(self.__input_dm)
            else:
                raise ("Error: input data must be either numpy.ndarray or pandas.DataFrame.")


    def write_out_ragged(self, quantized):
        '''
        Helper function that writes out pd.DataFrame to file in the \
        temporary directory for the class

        Inputs -
            quantized (boolean): use quantized data or original data

        Outputs -
            (None)
        '''

        if quantized:
            input_data = self.__quantized_input
        else:
            input_data = self.__input_dm

        if isinstance(input_data, pd.DataFrame):
            data = input_data.values.tolist()
        elif isinstance(input_data, np.ndarray):
            data = input_data.tolist()
        else:
            raise TypeError("Error: input data must be either numpy.ndarray or pandas.DataFrame.")

        to_write = []
        for row in data:
            to_write.append( [int(x) for x in row if not pd.isnull(x)] )

        self.__input_dm_fh = tempfile.NamedTemporaryFile(dir=self.__file_dir, delete=False)
        wr = csv.writer(self.__input_dm_fh, delimiter=" ")
        wr.writerows(to_write)
        self.__input_dm_fh.close()
        self.__input_dm_fname = path_leaf(self.__input_dm_fh.name)


    def run_dm(self, quantized, first_run_dm, max_len=None, num_run_dms=10, details=False):
        '''
        Helper function to call bin/smash to compute the distance matrix on the given input
        timeseries and write I/O files necessary for smash

        Inputs -
            ml (int): max length of data to use
            nr (int): number of runs of smashmatch used to create distance matrix
            d (boolean): do or do not show cpu usage of smashing algorithms while they run

        Outuputs -
            (numpy.ndarray) distance matrix of the input timeseries (shape n_samples x n_samples)
        '''

        if not first_run_dm:
            os.unlink(self.__input_dm_fh.name)
            self.__command = (self.bin_path + "/smash")

        if not quantized:
            self.write_out_ragged(False)
        else:
            self.write_out_ragged(True)

        self.__command += " -f " + self.__input_dm_fname + " -D row -T symbolic"

        if max_len is not None:
            self.__command += (" -L " + str(max_len))
        if num_run_dms is not None:
            self.__command += (" -n " + str(num_run_dms))
        if not details:
            self.__command += (" -t 0")

        self.__output_dm_fname = str(uuid.uuid4())
        self.__output_dm_fname = self.__output_dm_fname.replace("-", "")
        self.__command += (" -o " + self.__output_dm_fname)

        prev_wd = os.getcwd()
        os.chdir(self.__file_dir)
        sp.Popen(self.__command, shell=True).wait()
        os.chdir(prev_wd)

        try:
            results = np.loadtxt(fname=(self.__file_dir +"/"+self.__output_dm_fname), dtype=float)
            return results
        except IOError:
            print "Error: Smash calculation unsuccessful. Please try again."


    def fit(self, ml=None, nr=None, d=False):
        '''
        Uses Data Smashing to compute the distance matrix of the input time series

        Inputs -
            ml (int): max length of data to use
            nr (int): number of runs of smashmatch used to create distance matrix
            d (boolean): do or do not show cpu usage of smashing algorithms while they run

        Outuputs -
            (numpy.ndarray) distance matrix of the input timeseries (shape n_samples x n_samples)
        '''

        if self.__quantized_input is None:
            if isinstance(self.__input_dm, pd.DataFrame):
                floats_df = self.__input_dm.select_dtypes(include=[np.float])
                if floats_df.size > 0:
                    raise ValueError("Error: input to Smashing algorithm cannot be of type float; \
                    data not properly quantized .")
                else: # will run_dm smash on self.__input_dm, which are all ints
                    if self.__input_dm_fh is None:
                        return self.run_dm(False, True, ml, nr, d)
                    else:
                        return self.run_dm(False, False, ml, nr, d)
            elif isinstance(self.__input_dm, np.ndarray):
                if np.issubdtype(self.__input_dm.dtype, float):
                    raise ValueError("Error: input to Smashing algorithm cannot be of type float; \
                    data not properly quantized .")
                else:
                    if self.__input_dm_fh is None:
                        return self.run_dm(False, True, ml, nr, d)
                    else:
                        return self.run_dm(False, False, ml, nr, d)
            else:
                raise TypeError("Error: input data must be either numpy.ndarray or pandas.DataFrame.")
        else:
            if isinstance(self.__quantized_input, np.ndarray) or isinstance(self.__quantized_input, pd.DataFrame):
                if self.__input_fh is None:
                    return self.run_dm(True, True, ml, nr, d)
                else:
                    return self.run_dm(True, False, ml, nr, d)
            else:
                raise TypeError("Error: input data must be either numpy.ndarray or pandas.DataFrame.")


    def fit_predict(self, ml=None, nr=None, d=False):
        '''
        Returns sklearn fit_predict of distance matrix computed by data smashing algorithm
        and runs sklearn.cluster.KMeans clustering algorithm

        Inputs -
            ml (int): max length of data to use
            nr (int): number of runs of smashmatch used to create distance matrix
            d (boolean): do or do not show cpu usage of smashing algorithms while they run

        Returns -
            (np.ndarray) Compute cluster centers and predict cluster index from the input
            data using data smashing and sklearn.manifold.MDS
        '''

        self.__input_e = self.fit(ml, nr, d)
        return self.__KMeans_class.fit_predict(self.__input_e)


    def fit_transform(self, ml=None, nr=None, d=False):
        '''
        Returns sklearn fit_transform of distance matrix computed by data smashing algorithm
        and runs sklearn.cluster.KMeans clustering algorithm

        Inputs -
            ml (int): max length of data to use
            nr (int): number of runs of smashmatch used to create distance matrix
            d (boolean): do or do not show cpu usage of smashing algorithms while they run

        Returns -
            (np.ndarray) Compute cluster centers and predict cluster index from the input
            data using data smashing and sklearn.manifold.MDS
        '''

        self.__input_e = self.fit(ml, nr, d)
        return self.__KMeans_class.fit_transform(self.__input_e)


def cleanup():
    '''
    Clean up library files before closing the script; no I/O
    '''

    prev_wd = os.getcwd()
    os.chdir(cwd)
    if os.path.exists(cwd + "/" + temp_dir):
        command = "rm -r " + cwd + "/" + temp_dir
        sp.Popen(command, shell=True).wait()
    os.chdir(prev_wd)



atexit.register(cleanup)
