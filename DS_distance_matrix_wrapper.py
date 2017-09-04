import os, csv, pdb, tempfile, ntpath, uuid, atexit
import subprocess as sp
import numpy as np
from numpy import nan
import pandas as pd



# Potential known bug: does tempfile create NamedTemporaryFile names with hyphens?



# global variables
cwd = os.getcwd()
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



def q(x):
    if x <= 0:
        return 0
    else:
        return 1



class SmashDistancePrimitive:
    '''
    Object for running Smashmatch to calculate the distance matrix between time series

    Inputs -
        bin_path_(string): Path to smashmatch as a string
        quantiziation (function): quantization function for time series data

    Attributes:
        bin_path (string): path to bin/smash
    '''

    def __init__(self, bin_path_, X, quantizer_=None):
        self.__bin_path = bin_path_
        self.__quantizer = quantizer_
        self.__input = X
        prev_wd = os.getcwd()
        os.chdir(cwd)
        sp.Popen("mkdir "+ temp_dir, shell=True).wait()
        self.__file_dir = cwd + "/" + temp_dir
        os.chdir(prev_wd)
        if quantizer_ is not None:
            if isinstance(self.__input, pd.DataFrame):
                self.__quantized_input = self.__input.applymap(self.__quantizer)
            elif isinstance(self.__input, np.ndarray):
                self.__quantizer = np.vectorize(self.__quantizer)
                self.__quantized_input = self.__quantizer(self.__input)
            else:
                raise ("Error: input data must be either numpy.ndarray or pandas.DataFrame.")
        else:
            self.__quantized_input = None
        self.__input_fh = None
        self.__input_fname = None
        self.__out_fname = None
        self.__command = self.__bin_path


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


    @bin_path.setter
    def bin_path(self, new_bin_path):
        self.__bin_path = new_bin_path
        self.__command = self.__bin_path


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
            input_data = self.__input

        if isinstance(input_data, pd.DataFrame):
            data = input_data.values.tolist()
        elif isinstance(input_data, np.ndarray):
            data = input_data.tolist()
        else:
            raise TypeError("Error: input data must be either numpy.ndarray or pandas.DataFrame.")

        to_write = []
        for row in data:
            to_write.append( [int(x) for x in row if not pd.isnull(x)] )

        self.__input_fh = tempfile.NamedTemporaryFile(dir=self.__file_dir, delete=False)
        wr = csv.writer(self.__input_fh, delimiter=" ")
        wr.writerows(to_write)
        self.__input_fh.close()
        self.__input_fname = path_leaf(self.__input_fh.name)


    def run(self, quantized, first_run, max_len=None, num_runs=None,details=False):
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
        if not first_run:
            os.unlink(self.__input_fh.name)
            self.__command = self.__bin_path

        if not quantized:
            self.write_out_ragged(False)
        else:
            self.write_out_ragged(True)

        self.__command += " -f " + self.__input_fname + " -D row -T symbolic"

        if max_len is not None:
            self.__command += (" -L " + str(max_len))
        if num_runs is not None:
            self.__command += (" -n " + str(num_runs))
        if not details:
            self.__command += (" -t 0")

        self.__output_fname = str(uuid.uuid4())
        self.__output_fname = self.__output_fname.replace("-", "")
        self.__command += (" -o " + self.__output_fname)

        prev_wd = os.getcwd()
        os.chdir(self.__file_dir)
        sp.Popen(self.__command, shell=True).wait()
        os.chdir(prev_wd)

        try:
            results = np.loadtxt(fname=(self.__file_dir +"/"+self.__output_fname), dtype=float)
            return results
        except IOError:
            print "Error: Smash calculation unsuccessful. Please try again."
            return None


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
            if isinstance(self.__input, pd.DataFrame):
                floats_df = self.__input.select_dtypes(include=[np.float])
                if floats_df.size > 0:
                    raise ValueError("Error: input to Smashing algorithm cannot be of type float; \
                    data not properly quantized .")
                else: # will run smash on self.__input, which are all ints
                    if self.__input_fh is None:
                        return self.run(False, True, ml, nr, d)
                    else:
                        return self.run(False, False, ml, nr, d)
            elif isinstance(self.__input, np.ndarray):
                if np.issubdtype(self.__input.dtype, float):
                    raise ValueError("Error: input to Smashing algorithm cannot be of type float; \
                    data not properly quantized .")
                else:
                    if self.__input_fh is None:
                        return self.run(False, True, ml, nr, d)
                    else:
                        return self.run(False, False, ml, nr, d)
            else:
                raise TypeError("Error: input data must be either numpy.ndarray or pandas.DataFrame.")
        else:
            if isinstance(self.__quantized_input, np.ndarray) or isinstance(self.__quantized_input, pd.DataFrame):
                if self.__input_fh is None:
                    return self.run(True, True, ml, nr, d)
                else:
                    return self.run(True, False, ml, nr, d)
            else:
                raise TypeError("Error: input data must be either numpy.ndarray or pandas.DataFrame.")


    def set_input(self, new_input, new_quantizer=None):
        '''
        Setter method for class input attributes; modifies object in place
        No I/O
        '''
        self.__input = new_input
        if new_quantizer is not None:
            self.__quantizer = new_quantizer
            if isinstance(self.__input, pd.DataFrame):
                self.__quantized_input = self.__input.applymap(self.__quantizer)
            elif isinstance(self.__input, np.ndarray):
                self.__quantizer = np.vectorize(self.__quantizer)
                self.__quantized_input = self.__quantizer(self.__input)
            else:
                raise ("Error: input data must be either numpy.ndarray or pandas.DataFrame.")



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
