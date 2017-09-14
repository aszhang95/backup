import os
import csv
import pdb
import tempfile
import ntpath
import uuid
import atexit
import sys
import subprocess as sp
import numpy as np
from numpy import nan
import pandas as pd
from unsupervisedSeriesLearningPrimitiveBase import *
from seriesUtil import write_series



# Potential known bug: does tempfile create NamedTemporaryFile names with hyphens?


# global variables
CWD = os.getcwd()
TEMP_DIR = str(uuid.uuid4())
TEMP_DIR = TEMP_DIR.replace("-", "")




class SmashFeaturization(UnsupervisedSeriesLearningBase):
    '''
    Object for running Data Smashing to calculate the distance matrix between
        timeseries and using Sippl and/or sklearn.manifold to extract features;
        inherits from the UnsupervisedSeriesLearningBase API

    Inputs -
        bin_path(string): Path to Data Smashing binary as a string
        input_class (Input Object): Input data
        n_feats (int): number of dimensions for featurization algorithm
        feat_class (class): custom class to featurize input timeseries

    Attributes:
        bin_path (string): path to bin/smash
        num_features (int): number of dimensions to use for featurization
        featurization_class (class): custom class to featurize input timeseries
    '''

    def __init__(self, bin_path, input_class, n_feats=2, feat_class=None):
        self.__bin_path = os.path.abspath(bin_path)
        self.__input_class = input_class
        self._data = self.__input_class.data
        self.__num_features = n_feats
        self.__featurization_class = feat_class
        prev_wd = os.getcwd()
        os.chdir(CWD)
        sp.Popen("mkdir "+ TEMP_DIR, shell=True, stderr=sp.STDOUT).wait()
        self.__file_dir = CWD + "/" + TEMP_DIR
        os.chdir(prev_wd)
        self.__quantized_data  = self.__input_class.get()
        self._problem_type = "featurization"
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
    def num_features(self):
        return self.__num_features


    @property
    def featurization_class(self):
        return self.__featurization_class


    @property
    def data(self):
        return self._data


    @bin_path.setter
    def bin_path(self, new_bin_path):
        self.__bin_path = os.path.abspath(new_bin_path)
        self.__command = self.__bin_path


    @num_features.setter
    def num_features(self, new_ndim):
        assert isinstance(new_ndim, int), \
        "Error: number of features must be an int."
        self.__num_features = new_ndim


    @featurization_class.setter
    def featurization_class(self, new_feat_class):
        self.__featurization_class = new_feat_class


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
            self.__input_dm_fh, self.__input_dm_fname = write_series(self._data)
        else:
            self.__input_dm_fh, self.__input_dm_fname = write_series(self.__quantized_data)

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
            results = np.loadtxt(fname=\
            (self.__file_dir +"/"+self.__output_dm_fname), dtype=float)
            return results
        except IOError:
            print "Error: Smash calculation unsuccessful. Please try again."


    def fit_asymmetric(self, y=None, init=None):
        '''
        Helper method:
        Transforms distance matrix data to be symmetric and compatible with
        sklearn parameter specifications

        Inputs -
            y and init (numpy.ndarray): parameters of sklearn fit methods

        Returns -
            (None): modifies instance in place
        '''

        self._output = self._output.astype(np.float64)
        try:
            self.__featurization_class.fit(self._output, y, init)
        except ValueError:
            self._output = self._output + self._output.T
            try:
                self.__featurization_class.fit_transform(self._output, y,\
                init)
            except:
                warnings.warn(\
                "featurization error: Unable to fit. \
                Please ensure input featurization class takes\
                distance matrices as input.")


    def fit(self, ml=None, nr=None, d=False, y=None, init=None):
        '''
        Uses Data Smashing to compute the distance matrix of the timeseries;
            if an featurization glass has been defined, then fit on that
            featurization class will be run on the distance matrix

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
                    if self.__featurization_class is not None:
                        self.fit_asymmetric(y, init)
                    return self._output
                else:
                    self._output = self.run_dm(False, False, ml, nr, d)
                    if self.__featurization_class is not None:
                        self.fit_asymmetric(y, init)
                    return self._output
        else:
            if self.__input_dm_fh is None:
                self._output = self.run_dm(True, True, ml, nr, d)
                if self.__featurization_class is not None:
                    self.fit_asymmetric(y, init)
                return self._output
            else:
                self._output = self.run_dm(True, False, ml, nr, d)
                if self.__featurization_class is not None:
                    self.fit_asymmetric(y, init)
                return self._output


    def fit_transform(self, ml=None, nr=None, d=False, \
    featurizer='default', y=None, init=None):
        '''
        Computes Data Smashing distance matrix and returns
        the resulting featurized coordinates

        Inputs -
            ml (int): max length of data to use
            nr (int): number of runs of Smash to compute distance matrix
                (refines results)
            d (boolean): do (True) or do not (False) show cpu usage of Smash
                algorithm
            featurizer (string) either 'default' to use Sippl featurization or
                'custom' to use user-defined emebedding class
            y and init (numpy.ndarray): parameters for the fit_transform method
                of sklearn featurization classes

        Outuputs -
            (numpy.ndarray) the featurized coordinates of the input data
            (shape num_timeseries x num_features)
        '''

        self.__input_e = self.fit(ml, nr, d)
        # since you run fit, the assumption can read in from the file you wrote out to in fit
        # i.e. self.__output_dm_fname is the name for the input

        if featurizer == 'default':
            prev_wd = os.getcwd()
            os.chdir(self.__file_dir)
            command = (self.__bin_path + "/embed -f ")

            if os.path.isfile(self.__output_dm_fname):
                command += self.__output_dm_fname
            else: # should be impossible
                print(\
                "Smash featurization encountered an error. Please try again.")
                sys.exit(1)

            if not d:
                FNULL = open(os.devnull, 'w')
                sp.Popen(command, shell=True, stdout=FNULL, stderr=sp.STDOUT,\
                 close_fds=True).wait()
            else:
                sp.Popen(command, shell=True, stderr=sp.STDOUT).wait()

            try:
                sippl_feats = np.loadtxt(fname="outE.txt", dtype=float)
                if self.__num_features > sippl_feats.shape[1]:
                    raise ValueError("Error: Number of dimensions specified \
                    greater than dimensions of input data")
                sippl_feat = sippl_feats[:, :self.__num_features]
                os.chdir(prev_wd)
                self._output = sippl_feat
                return self._output
            except IOError or IndexError:
                print "Error: featurization unsuccessful. Please try again."
        else:
            self.__input_e = self.__input_e.astype(np.float64)
            try:
                self._output = self.__featurization_class.fit_transform(\
                self.__input_e, y, init)
                return self._output

            except ValueError:
                self.__input_e = self.__input_e + self.__input_e.T
                try:
                    self._output = self.__featurization_class.fit_transform(\
                    self.__input_e, y, init)
                    return self._output
                except:
                    warnings.warn(\
                    "featurization error: \
                    please ensure input featurization class takes\
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


    def transform(self, ml=None, nr=None, d=False, \
    featurizer='default', init=None):
        '''
        Returns the resulting featurized coordinates
            from the fitted featurization class

        Inputs -
            ml (int): max length of data to use
            nr (int): number of runs of Smash to compute distance matrix
                (refines results)
            d (boolean): do (True) or do not (False) show cpu usage of Smash
                algorithm
            featurizer (string) either 'default' to use Sippl featurization or
                'custom' to use user-defined emebedding class
            y and init (numpy.ndarray): parameters for the fit_transform method
                of sklearn.featurization classes

        Outuputs -
            (numpy.ndarray) the featurized coordinates of the input data
            (shape num_timeseries x num_features)
        '''

        assert self._output is not None, \
        "Error: no featurization class has been fit"
        if featurizer == 'default':
            prev_wd = os.getcwd()
            os.chdir(self.__file_dir)
            command = (self.__bin_path + "/embed -f ")

            if os.path.isfile(self.__output_dm_fname):
                command += self.__output_dm_fname
            else: # should be impossible
                print(\
                "Smash featurization encountered an error. Please try again.")
                sys.exit(1)

            if not d:
                FNULL = open(os.devnull, 'w')
                sp.Popen(command, shell=True, stdout=FNULL, stderr=sp.STDOUT,\
                 close_fds=True).wait()
            else:
                sp.Popen(command, shell=True, stderr=sp.STDOUT).wait()

            try:
                sippl_feats = np.loadtxt(fname="outE.txt", dtype=float)
                if self.__num_features > sippl_feats.shape[1]:
                    raise ValueError("Error: Number of dimensions specified \
                    greater than dimensions of input data")
                sippl_feat = sippl_feats[:, :self.__num_features]
                os.chdir(prev_wd)
                self._output = sippl_feat
                return self._output
            except IOError or IndexError:
                print "Error: featurization unsuccessful. Please try again."
        else:
            self._ouput = self._ouput.astype(np.float64)
            try:

                try:
                    y_ = self._data_additional.data
                except AttributeError:
                    y_ = None

                self._output = self.__featurization_class.transform(\
                self._ouput, y_, init)
                return self._output

            except ValueError:
                self._ouput = self._ouput + self._ouput.T

                try:
                    y_ = self._data_additional.data
                except AttributeError:
                    y_ = None

                try:
                    self._output = self.__featurization_class.transform(\
                    self._ouput, y_, init)
                    return self._output
                except:
                    warnings.warn(\
                    "featurization error: \
                    Please ensure input featurization class takes\
                    distance matrices as input.")
                    return None



def cleanup():
    '''
    Maintenance function:
    Clean up library files before closing the script; no I/O
    '''

    prev_wd = os.getcwd()
    os.chdir(CWD)
    if os.path.exists(CWD + "/" + TEMP_DIR):
        command = "rm -r " + CWD + "/" + TEMP_DIR
        sp.Popen(command, shell=True, stderr=sp.STDOUT).wait()
    os.chdir(prev_wd)



atexit.register(cleanup)
