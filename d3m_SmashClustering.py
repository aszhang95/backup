import os, csv, pdb, tempfile, ntpath, uuid, atexit, sys
import subprocess as sp
import numpy as np
from numpy import nan
import pandas as pd
from sklearn import cluster
from d3m_unsup_wo_output_class import *



# Potential known bug: does tempfile create NamedTemporaryFile names with hyphens?


# global variables
cwd = os.getcwd()
temp_dir = str(uuid.uuid4())
temp_dir = temp_dir.replace("-", "")



class SmashClustering(Unsupervised_Series_Learning_Base):
    '''
    Object for run_dmning Smashmatch to calculate the distance matrix between time series
    and using Sippl and/or sklearn.manifold.MDS to embed

    Inputs -
        bin_path_(string): Path to smashmatch as a string
        quantiziation (function): quantization function for time series data

    Attributes:
        bin_path (string): path to bin/smash
        quantizer (function): function to quantify input data
        quantized input (numpy.ndarray or pd.DataFrame) the transformed input data if
            quantizer function is specified
        num_dimensions (int): number of dimensions used for embedding

        (Note: bin_path and num_dimensions can be set by assignment, input and quantizer must be
            set using custom method)
    '''

    def __init__(self, bin_path_, input_class_, n_clus, cluster_class=None):
        self.__bin_path = os.path.abspath(bin_path_)
        self.__input_class = input_class_
        self._data = self.__input_class.data
        self.__num_clusters = n_clus
        if cluster_class is None:
            self.__cluster_class = cluster.KMeans(n_clusters=self.__num_clusters)
        else:
            self.__cluster_class = cluster_class
        prev_wd = os.getcwd()
        os.chdir(cwd)
        sp.Popen("mkdir "+ temp_dir, shell=True, stderr=sp.STDOUT).wait()
        self.__file_dir = cwd + "/" + temp_dir
        os.chdir(prev_wd)
        self.__quantized_data  = self.__input_class.get()
        self._problem_type = "clusering"
        self.__input_dm_fh = None
        self.__input_dm_fname = None
        self.__output_dm_fname = None
        self.__command = (self.__bin_path + "/smash")
        self._output = None
        self.__input_e = None


    @property
    def bin_path(self):
        return self.__bin_path


    @property
    def num_clusters(self):
        return self.__num_clusters


    @bin_path.setter
    def bin_path(self, new_bin_path):
        self.__bin_path = os.path.abspath(new_bin_path)
        self.__command = self.__bin_path


    @num_clusters.setter
    def num_clusters(self, new_nclus):
        assert isinstance(new_nclus, int), "Error: num_clusters must be an int."
        self.__num_clusters = new_nclus


    @property
    def data(self):
        return self._data


    @data.setter
    def data(self, input_data):
        if not isinstance(input_data,Input):
            raise Exception('data must be instance of Input class')
        self.__input_class = input_data
        self._data = self.__input_class.data
        self.__quantized_data = self.__input_class.get()


    @property
    def cluster_class(self):
        return self.__cluster_class


    @cluster_class.setter
    def cluster_class(self, cc):
        self.__cluster_class = cc


    def path_leaf(self, path):
        '''
        Helper function:
        Returns filename from a given path/to/file
        Taken entirely from Lauritz V. Thaulow on https://stackoverflow.com/questions/8384737

        Input -
            path (string): path/to/the/file

        Returns -
            filename (string)
        '''

        head, tail = ntpath.split(path)
        return tail or ntpath.basename(head)


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
            input_data = self.__quantized_data
        else:
            input_data = self._data

        data = input_data.tolist()

        to_write = []
        for row in data:
            to_write.append( [int(x) for x in row if not np.isnan(x)] )

        self.__input_dm_fh = tempfile.NamedTemporaryFile(dir=self.__file_dir, delete=False)
        wr = csv.writer(self.__input_dm_fh, delimiter=" ")
        wr.writerows(to_write)
        self.__input_dm_fh.close()
        self.__input_dm_fname = self.path_leaf(self.__input_dm_fh.name)


    def run_dm(self, quantized, first_run_dm, max_len=None, num_run_dms=5, details=False):
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
            self.__command = (self.__bin_path + "/smash")

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
        sp.Popen(self.__command, shell=True, stderr=sp.STDOUT).wait()
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

        if self.__quantized_data is None: # no checks because assume data was pre-processed in Input
            if np.issubdtype(self._data .dtype, float):
                raise ValueError("Error: input to Smashing algorithm cannot be of type float; \
                data not properly quantized .")
            else:
                if self.__input_dm_fh is None:
                    self._output = self.run_dm(False, True, ml, nr, d)
                    return self._output
                else:
                    self._output = self.run_dm(False, False, ml, nr, d)
                    return self._output
        else:
            if self.__input_dm_fh is None:
                self._output = self.run_dm(True, True, ml, nr, d)
                return self._output
            else:
                self._output = self.run_dm(True, False, ml, nr, d)
                return self._output


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
        self._output = self.__cluster_class.fit_predict(self.__input_e)
        return self._output


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
        self._output = self.__cluster_class.fit_transform(self.__input_e)
        return self._output


    def predict(self, ml=None, nr=None, d=False):
        self.__input_e = self.fit(ml, nr, d)
        self._output = self.__cluster_class.predict(self.__input_e)
        return self._output


    def predict_proba(self,*arg,**kwds):
        warnings.warn('Warning: predict_proba method for this class is undefined.')
        pass


    def log_proba(self,*arg,**kwds):
        warnings.warn('Warning: log_proba method for this class is undefined.')
        pass


    def score(self, ml=None, nr=None, d=False):
        self.__input_e = self.fit(ml, nr, d)
        self._output = self.__cluster_class.score(self.__input_e)
        return self._output


    def transform(self, ml=None, nr=None, d=False):
        self.__input_e = self.fit(ml, nr, d)
        self._output = self.__cluster_class.transform(self.__input_e)
        return self._output


def cleanup():
    '''
    Clean up library files before closing the script; no I/O
    '''

    prev_wd = os.getcwd()
    os.chdir(cwd)
    if os.path.exists(cwd + "/" + temp_dir):
        command = "rm -r " + cwd + "/" + temp_dir
        sp.Popen(command, shell=True, stderr=sp.STDOUT).wait()
    os.chdir(prev_wd)



atexit.register(cleanup)
