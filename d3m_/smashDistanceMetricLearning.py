import os
import pdb
import uuid
import atexit
import sys
import subprocess as sp
import numpy as np
from numpy import nan
import pandas as pd
from primitives_interfaces.unsupervised_learning_series_modeling import *
from primitives_interfaces.utils.series import write_series



# global variables
CWD = os.getcwd()
TEMP_DIR = str(uuid.uuid4())
TEMP_DIR = TEMP_DIR.replace("-", "")



class SmashDistanceMetricLearning(UnsupervisedSeriesLearningBase):
    '''
    Object for running Data Smashing  to calculate the distance matrix between
    multiple timeseries; Inherits from UnsupervisedSeriesLearningBase API

    Inputs -
        bin_path(string): Path to Smash binary as a string
        input_class (Input Object): Input data

    Attributes:
        bin_path (string): path to bin/smash
    '''

    def __init__(self, bin_path, input_class):
        self.__bin_path = os.path.abspath(bin_path)
        self.__input_class = input_class
        self._data = self.__input_class.data
        prev_wd = os.getcwd()
        os.chdir(CWD)
        sp.Popen("mkdir "+ TEMP_DIR, shell=True, stderr=sp.STDOUT).wait()
        self.__file_dir = CWD + "/" + TEMP_DIR
        os.chdir(prev_wd)
        self.__quantized_data  = self.__input_class.get()
        self._problem_type = "distance_metric_learning"
        self.__input_dm_fh = None
        self.__input_dm_fname = None
        self.__output_dm_fname = None
        self.__command = (self.__bin_path + "/smash")
        self._output = None


    @property
    def bin_path(self):
        return self.__bin_path


    @bin_path.setter
    def bin_path(self, new_bin_path):
        self.__bin_path = os.path.abspath(new_bin_path)
        self.__command = self.__bin_path


    @property
    def data(self):
        return self._data


    @property
    def quantized_data(self):
        return self.__quantized_data


    @data.setter
    def data(self, input_data):
        if not isinstance(input_data,Input):
            raise Exception('data must be instance of Input class')
        self.__input_class = input_data
        self._data = self.__input_class.data
        self.__quantized_data = self.__input_class.get()


    # for interfacing with util functions
    @property
    def file_dir(self):
        return self.__file_dir


    def get_dm(self, quantized, first_run, max_len=None, \
    num_get_dms=5, details=False):
        '''
        Helper function:
        Calls bin/smash to compute the distance matrix on the given input
        timeseries and write I/O files necessary for Data Smashing

        Inputs -
            max_len (int): max length of data to use
            num_get_dms (int): number of runs of Smash to compute distance
                matrix (refines results)
            details (boolean): do (True) or do not (False) show cpu usage of
                Data Smashing algorithm

        Outuputs -
            (numpy.ndarray) distance matrix of the input timeseries
            (shape n_samples x n_samples)
        '''

        if not first_run:
            os.unlink(self.__input_dm_fh.name)
            self.__command = (self.__bin_path + "/smash")

        if not quantized:
            self.__input_dm_fh, self.__input_dm_fname = write_series(input_data=self._data,\
                                                                    file_dir=self.__file_dir)
        else:
            self.__input_dm_fh, self.__input_dm_fname = write_series(input_data=self.__quantized_data,\
                                                                    file_dir=self.__file_dir)

        self.__command += " -f " + self.__input_dm_fname + " -D row -T symbolic"

        if max_len is not None:
            self.__command += (" -L " + str(max_len))
        if num_get_dms is not None:
            self.__command += (" -n " + str(num_get_dms))
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
                "Error: input to Smashing algorithm cannot be of type float; \
                data not properly quantized .")
            else:
                if self.__input_dm_fh is None:
                    self._output = self.get_dm(False, True, ml, nr, d)
                    return self._output
                else:
                    self._output = self.get_dm(False, False, ml, nr, d)
                    return self._output
        else:
            if self.__input_dm_fh is None:
                self._output = self.get_dm(True, True, ml, nr, d)
                return self._output
            else:
                self._output = self.get_dm(True, False, ml, nr, d)
                return self._output


    def fit_transform(self,*arg,**kwds):
        warnings.warn(\
        'Warning: fit_transform method for this class is undefined.')
        pass


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
    os.chdir(CWD)
    if os.path.exists(CWD + "/" + TEMP_DIR):
        command = "rm -r " + CWD + "/" + TEMP_DIR
        sp.Popen(command, shell=True, stderr=sp.STDOUT).wait()
    os.chdir(prev_wd)



atexit.register(cleanup)
