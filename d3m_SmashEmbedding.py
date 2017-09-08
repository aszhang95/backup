import os, csv, pdb, tempfile, ntpath, uuid, atexit, sys
import subprocess as sp
import numpy as np
from numpy import nan
import pandas as pd
from d3m_unsup_wo_output_class import *



# Potential known bug: does tempfile create NamedTemporaryFile names with hyphens?


# global variables
cwd = os.getcwd()
temp_dir = str(uuid.uuid4())
temp_dir = temp_dir.replace("-", "")




class SmashEmbedding(Unsupervised_Series_Learning_Base):
    '''
    Object for running Data Smashing to calculate the distance matrix between
        timeseries and using Sippl and/or sklearn.manifold.MDS to embed;
        inherits from the Unsupervised_Series_Learning_Base API

    Inputs -
        bin_path_(string): Path to Data Smashing binary as a string
        input_class_ (Input Object): Input data
        n_dim (int): number of dimensions for embedding algorithm

    Attributes:
        bin_path (string): path to bin/smash
        num_dimensions (int): number of dimensions to use for embedding
    '''

    def __init__(self, bin_path_, input_class_, n_dim):
        self.__bin_path = os.path.abspath(bin_path_)
        self.__input_class = input_class_
        self._data = self.__input_class.data
        self.__num_dimensions = n_dim
        prev_wd = os.getcwd()
        os.chdir(cwd)
        sp.Popen("mkdir "+ temp_dir, shell=True, stderr=sp.STDOUT).wait()
        self.__file_dir = cwd + "/" + temp_dir
        os.chdir(prev_wd)
        self.__quantized_data  = self.__input_class.get()
        self._problem_type = "embedding"
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
    def num_dimensions(self):
        return self.__num_dimensions


    @bin_path.setter
    def bin_path(self, new_bin_path):
        self.__bin_path = os.path.abspath(new_bin_path)
        self.__command = self.__bin_path


    @num_dimensions.setter
    def num_dimensions(self, new_ndim):
        assert isinstance(new_ndim, int), \
        "Error: num_dimensions must be an int."
        self.__num_dimensions = new_ndim


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


    def path_leaf(self, path):
        '''
        Helper function:
        Returns filename from a given path/to/file
        Taken entirely from Lauritz V. Thaulow on
        https://stackoverflow.com/questions/8384737

        Input -
            path (string): path/to/the/file

        Returns -
            filename (string)
        '''

        head, tail = ntpath.split(path)
        return tail or ntpath.basename(head)


    def write_out_ragged(self, quantized):
        '''
        Helper function:
        Writes out input numpy.ndarray to file to interface with Data Smashing
            binary

        Inputs -
            quantized (boolean): use quantized data (True) or original data
                (False)

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


    def run_dm(self, quantized, first_run_dm, max_len=None, \
    num_run_dms=5, details=False):
        '''
        Helper function:
        Calls bin/smash to compute the distance matrix on the given input
        timeseries and write I/O files necessary for Data Smashing

        Inputs -
            max_len (int): max length of data to use
            num_run_dms (int): number of runs of Smash to compute distance
                matrix (refines results)
            details (boolean): do (True) or do not (False) show cpu usage of
                Data Smashing algorithm

        Outuputs -
            (numpy.ndarray) distance matrix of the input timeseries
            (shape n_samples x n_samples)
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
        Uses Data Smashing to compute the distance matrix of the timeseries

        Inputs -
            ml (int): max length of data to use
            nr (int): number of runs of Smash to compute distance matrix
                (refines results)
            d (boolean): do (True) or do not (False) show cpu usage of Data
            Smashing algorithm

        Outuputs -
            (numpy.ndarray) distance matrix of the input timeseries
            (shape n_samples x n_samples)
        '''

        if self.__quantized_data is None: # no checks because assume data was pre-processed in Input
            if np.issubdtype(self._data .dtype, float):
                raise ValueError(\
                "Error: input to Smashing algorithm cannot be of type float;\
                data not properly quantized.")
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


    def fit_transform(self, ml=None, nr=None, d=False, \
    embedder=None, init=None):
        '''
        Computes Data Smashing distance matrix and returns
        the resulting embedded coordinates

        Inputs -
            ml (int): max length of data to use
            nr (int): number of runs of Smash to compute distance matrix
                (refines results)
            d (boolean): do (True) or do not (False) show cpu usage of Smash
                algorithm
            embedder (instance of embedding class that embeds distance matrices
                with fit_transform function) e.g. sklearn.manifold.MDS
            y and init (numpy.ndarray): parameters for the fit_transform method
                of sklearn.embedding classes

        Outuputs -
            (numpy.ndarray) the embedded coordinates of the input data
            (shape num_timeseries x num_dimensions)
        '''

        self.__input_e = self.fit(ml, nr, d)
        # since you run fit, the assumption can read in from the file you wrote out to in fit
        # i.e. self.__output_dm_fname is the name for the input

        if embedder is None:
            prev_wd = os.getcwd()
            os.chdir(self.__file_dir)
            command = (self.__bin_path + "/embed -f ")

            if os.path.isfile(self.__output_dm_fname):
                command += self.__output_dm_fname
            else: # should be impossible
                print("Smash Embedding encountered an error. Please try again.")
                sys.exit(1)

            if not d:
                FNULL = open(os.devnull, 'w')
                sp.Popen(command, shell=True, stdout=FNULL, stderr=sp.STDOUT,\
                 close_fds=True).wait()
            else:
                sp.Popen(command, shell=True, stderr=sp.STDOUT).wait()

            try:
                sippl_embed = np.loadtxt(fname="outE.txt", dtype=float)
                if self.__num_dimensions > sippl_embed.shape[1]:
                    raise ValueError("Error: Number of dimensions specified \
                    greater than dimensions of input data")
                sippl_feat = sippl_embed[:, :self.__num_dimensions]
                os.chdir(prev_wd)
                self._output = sippl_feat
                return self._output
            except IOError or IndexError:
                print "Error: Embedding unsuccessful. Please try again."
        else:
            self.__input_e = self.__input_e.astype(np.float64)
            try:

                try:
                    y_ = self._data_additional.data
                except AttributeError:
                    y_ = None

                self._output = embedder.fit_transform(self.__input_e, y_, init)
                return self._output

            except ValueError:
                self.__input_e = self.__input_e + self.__input_e.T

                try:
                    y_ = self._data_additional.data
                except AttributeError:
                    y_ = None

                try:
                    self._output = embedder.fit_transform(self.__input_e, y_,\
                    init)
                    return self._output
                except:
                    warnings.warn(\
                    "Embedding error: please ensure input embedding class takes\
                    distance matrices as input.")
                    return None


    def fit_predict(self,*arg,**kwds):
        warnings.warn(\
        'Warning: fit_predict method for this class is undefined.')
        pass


    def predict(self,*arg,**kwds):
        warnings.warn(\
        'Warning: predict method for this class is undefined.')
        pass


    def predict_proba(self,*arg,**kwds):
        warnings.warn(\
        'Warning: predict_proba method for this class is undefined.')
        pass


    def log_proba(self,*arg,**kwds):
        warnings.warn(\
        'Warning: log_proba method for this class is undefined.')
        pass


    def score(self,*arg,**kwds):
        warnings.warn('Warning: score method for this class is undefined.')
        pass


    def transform(self,*arg,**kwds):
        warnings.warn('Warning: transform method for this class is undefined.')
        pass



def cleanup():
    '''
    Maintenance function:
    Clean up library files before closing the script; no I/O
    '''

    prev_wd = os.getcwd()
    os.chdir(cwd)
    if os.path.exists(cwd + "/" + temp_dir):
        command = "rm -r " + cwd + "/" + temp_dir
        sp.Popen(command, shell=True, stderr=sp.STDOUT).wait()
    os.chdir(prev_wd)



atexit.register(cleanup)
