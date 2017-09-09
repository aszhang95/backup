import os
import sys
import time
import csv
import math
import uuid
import atexit
import pdb
import warnings
import subprocess as sp
import numpy as np
from numpy import nan
import pandas as pd
import supervisedLearningPrimitiveBase
from seriesUtil import apply



# Global Variables: (ensures safe R/W of smashmatch)
PREFIX = str(uuid.uuid4())
PREFIX = PREFIX.replace("-", "")
TEMP_DIR = str(uuid.uuid4())
CWD = os.getcwd()
# PREFIX = "resx" # for testing purposes only



class LibFile:
    '''
    Helper class to store library file information to interface with
    file-I/O data smashing code

    Attributes -
        class (int):  label given to the class by SmashMatch
            based on the order in which it was fed in
        label (int): actual class label set by user
        filename (string): path to temporary library file
    '''

    def __init__(self, class_, label_, fname_):
        self.class_name = class_
        self.label = label_
        self.filename = fname_



class SmashMatchClassification(supervisedLearningPrimitiveBase.SupervisedLearningPrimitiveBase):
    '''
    Class for SmashMatch-based classification; I/O modeled after
    sklearn.SVM.SVC classifier and inherits from
     SupervisedLearningPrimitiveBase

    Inputs -
        bin_path(string): Path to smashmatch as a string
        preproc (function): function to quantize timeseries;

    Attributes:
        bin_path(string): Path to smashmatch as a string
        classes (np.1Darray): class labels fitted into the model;
            also column headers for predict functions
            NOTE: cannot be changed once fitted - rerun fit to use new classes
        preproc (function): quantization function for timeseries data
    '''

    # lib_files = list of library file names or LibFile Objects
    def __init__(self, bin_path, preproc=None):
        assert os.path.isfile(bin_path), "Error: invalid bin path."
        self.__bin_path = os.path.abspath(bin_path)
        prev_wd = os.getcwd()
        os.chdir(CWD)
        sp.Popen("mkdir "+ TEMP_DIR, shell=True, stderr=sp.STDOUT).wait()
        self.__file_dir = CWD + "/" + TEMP_DIR
        os.chdir(prev_wd)
        self.__classes = []
        self.__lib_files = [] # list of LibFile objects
        self.__lib_command = " -F "
        self.__command = self.__bin_path
        self.__input = None
        self.__mapper = {}
        if preproc is not None:
            self.__preproc = preproc


    @property
    def bin_path(self):
        return self.__bin_path


    @bin_path.setter
    def bin_path(self, new_path):
        assert os.path.isfile(new_path), "Error: invalid bin path."
        self.__bin_path = os.path.abspath(new_path)


    @property
    def classes(self):
        return self.__classes


    @property
    def preproc(self):
        return self.__preproc


    @preproc.setter
    def preproc(self, new_preproc):
        self.__preproc = new_preproc


    @property
    def state(self):
        warnings.warn('Warning: \
        SmashMatchClassification does not have attribute state:\
        SmashMatch does not classify based on models.')
        pass


    @state.setter
    def state(self, X):
        warnings.warn('Warning: \
        Cannot set state for SmashMatchClassification:\
        SmashMatch does not classify based on models.')
        pass


    ### helper functions for library files:
    def read_vert_series(self, directory, label=None, quantize=False):
        '''
        Helper method:
        Converts folder of timeseries with vertical orientation to
            pd.DataFrame for use with SmashMatch - takes all the files within
            the folder to be examples of a class and assumes each file has
            only one timeseries -> row within DataFrame
            NOTE: files (timeseries) can be of differing length;
            shorter time series will be padded with NaN's


        Inputs -
            directory (string): path to a directory or file
            label (char): class label
            quantize (boolean): if input timeseries have not been quantized,
                apply class quantizer if True

        Outputs -
            tuple of pandas.DataFrame with missing values as
            NaN and label if label provided otherwise only returns pd.DataFrame
        '''

        if os.path.isdir(directory):
            total = []
            max_cols = 0
            for filename in os.listdir(directory):
                row = []
                with open(directory+"/"+filename, "r") as f:
                    col_counter = 0
                    for line in f:
                        row.append(line.strip())
                        col_counter += 1
                if col_counter > max_cols:
                    max_cols = col_counter
                total.append(row)

            if not quantize:
                rv = pd.DataFrame(total, columns=range(max_cols), \
                dtype=np.int32)
                rv.fillna(value=nan, inplace=True)
            else:
                assert(self.__preproc is not None), \
                "Error: no quantization function defined"
                rv = pd.DataFrame(total, columns=range(max_cols), dtype=float)
                rv.fillna(value=nan, inplace=True)
                rv = apply(self.__preproc, rv)

            if label is not None:
                return (rv, label)
            else:
                return rv
        else: # path to a singular file
            row = []
            with open(directory, "r") as f:
                for line in f:
                    row.append(line.strip())

            if not quantize:
                rv = pd.DataFrame(total, dtype=np.int32)
            else:
                assert(self.__preproc is not None),\
                "Error: no quantization function defined"
                rv = pd.DataFrame(total, dtype=float)
                rv = apply(self.__preproc, rv)

            if label is not None:
                return (rv, label)
            else:
                return rv


    def read_series(self, filename, delimiter, quantize=False):
        '''
        Helper method:
        Reads in file (of potentially mixed line lengths)
            (timeseries of different length)(direction = horizontal)
            as a pandas.DataFrame where each row is a timeseries (from
            corresponding line); shorter timeseries padded with NaN's

        Inputs -
            filename (string): path to file
            delimiter (char): value delimiter
            quantize (boolean): if input timeseries have not been quantized,
                apply class quantizer if True

        Outputs -
            pandas.DataFrame with missing values as NaN
        '''

        data = []
        max_col_len = 0
        with open(filename, 'r') as f:
            for line in f:
                formatted_row = []
                row = line.strip().split(delimiter)
                if row[-1] == "":
                    row = row[:-1]
                row_len = len(row)
                if row_len > max_col_len:
                    max_col_len = row_len
                data.append(row)
            if not quantize:
                rv = pd.DataFrame(data, columns=range(max_col_len), dtype=np.int32)
                rv.fillna(value=nan, inplace=True)
            else:
                assert(self.__preproc is not None), "Error: no quantization function defined"
                rv = pd.DataFrame(data, columns=range(max_col_len), dtype=float)
                rv.fillna(value=nan, inplace=True)
                rv = apply(self.__preproc, rv)
        return rv


    def get_unique_name(self, lib):
        '''
        Helper method:
        Generates unique filename with the PREFIX "lib_" if the input
            file is a library files or no PREFIX if it is not a library file

        Inputs -
            lib (boolean) whether the generated name is for a library file (True)
            or for an input file (False)

        Outputs -
            unique_name
        '''
        rv = str(uuid.uuid4())
        rv = rv.replace("-", "")
        if lib:
            return "lib_" + rv
        else:
            return "input_" + rv


    def condense(self, mappings):
        '''
        Helper method:
        Creates X (timeseries), y (labels) necessary for SmashMatch following sklearn.SVM.SVC.fit method conventions

        Input -
            mappings(list of tuples of (df, label))

        Output -
            X (timeseries examples of each class, each row = unique timeseries)
            y (df with corresponding labels of n x 1, n=number of timeseries in X)
        '''

        labelled = []
        for mapping in mappings:
            class_ = mapping[1]
            nrows = mapping[0].shape[0]
            class_col = np.asarray([class_]*nrows)
            mapping[0].insert(0, 'type', class_col)
            labelled.append(mapping[0])

        combined = pd.concat(labelled)
        y_ = combined.pop("type")
        return combined, y_


    def make_libs_df(self, X, y):
        '''
        Helper method:
        Writes out class labels & examples to files usable by SmashMatch
            (delimited by spaces, each row is a timeseries, create a different file for each series)

        Inputs -
            X (pd.DataFrame): timeseries examples of each class, each row is a
                timeseries
            y (pd.Dataframe): labels for each timeseries

        Returns -
            lib_files (list of LibFile objects)
        '''

        if "level" not in X.columns:
            X.insert(0, "level", y)

        X.sort_values("level")

        labels = y.unique().tolist()
        class_num = 0

        lib_files = []
        for label_ in labels:
            df = X.loc[X.level == label_]
            df = df.drop("level", 1)
            fname = self.get_unique_name(True)
            self.write_out_series(fname, df)
            lib_files.append(LibFile(class_num, label_, fname))
            class_num += 1
        return lib_files


    def make_libs(self, X, y):
        '''
        Helper method:
        Writes out class labels & examples to files usable by SmashMatch
            (delimited by spaces, each row is a timeseries, create a different file for each series)

        Inputs -
            X (np.nda): timeseries examples of each class, each row is a
                timeseries
            y (np.1da): labels for each timeseries

        Returns -
            rv (list of LibFile objects)
        '''

        rv = []
        merged = np.c_[y, X] # merge to one large np array with labels at col 0
        merged = merged[np.argsort(merged[:, 0])] # sort in ascending order by col 0
        libs = np.split(merged, np.where(np.diff(merged[:,0]))[0]+1) # split ndas by class

        class_num = 0
        for class_ in libs:
            label_ = class_[0, 0]
            rows = []
            for stream in class_:
                rows.append(stream[1:].tolist()) # cut off label from row entry
            lib_f = self.get_lib_nameunique_name(True)
            with open(lib_f, "w") as f:
                wr = csv.writer(f, delimiter=" ")
                wr.writerows(rows)
            rv.append(LibFile(class_num, label_, lib_f))
            class_num += 1
        return rv


    def fit(self, X, y, quantize=False): # not sure what to do with kwargs or the classes/sample_weight params
        '''
        Reads in appropriate data/labels -> library class files
            to be used by SmashMatch
            (writes out library files in format usable by SmashMatch)

        Inputs -
            X (np.nda or pandas.DataFrame): class examples
            y (np.1da or pandas.Series): class labels
            quantize (boolean): if timeseries have not beeen quantized,
                apply instantiated quantizer to input timeseries examples

        Returns -
          (None) modifies object in place
        '''

        if quantize:
            assert(self.__preproc is not None), "Error: no quantization function defined"
            X = apply(self.__preproc, X)
        # delete old library files before running (would only be true after first run)
        len_libs = len(self.__lib_files)
        if len_libs != 0:
            self.clean_libs()

        if isinstance(X, np.ndarray):
            self.__lib_files = self.make_libs(X, y)
        elif isinstance(X, pd.DataFrame):
            self.__lib_files = self.make_libs_df(X, y)
        else:
            raise ValueError("Error: unsupported types for X. X can only be of type numpy.ndarray or pandas.DataFrame.")

        mappings = []
        # need to make sure we feed the class_names in according to their actual order
        for lib in self.__lib_files:
            mappings.append((lib.class_name, lib))
        mappings.sort(key=lambda x: x[0])
        for mapping in mappings:
            self.__mapper[mapping[0]] = mapping[1] # key = class_num, value = LibFile
            self.__classes.append(mapping[1].label)
            self.__lib_command += mapping[1].filename + ' ' # should return only the filename
        self.__classes = np.asarray(self.__classes)
        return


    def write_out_nda(self, array):
        '''
        Helper method:
        Writes out input data to file usable by SmashMatch (delimited by space)
            each row is a timeseries

        Inputs -
            array (np.nda): data to be classified; each row = unique timeseries

        Outputs -
            (string) filename of the input_file
        '''

        self.__input = array.astype(np.int32)
        rows = array.tolist()
        fname = self.get_unique_name(False)
        with open(self.__file_dir + "/" + fname, "w") as f:
            wr = csv.writer(input_fh, delimiter=" ")
            wr.writerows(rows)
        return fname


    def write_out_series(self, filename, df_):
        '''
        Helper method:
        Writes out pd.DataFrame to file to usable by SmashMatch (delimited by
            space, each line=unique timeseries)

        Inputs -
            filename (string): name of output file
            df (pd.DataFrame): timeseries to be written out

        Outputs -
            (None)
        '''
        assert(isinstance(df_, pd.DataFrame)), "Error: argument df is not type pandas.DataFrame"

        data = df_.values.tolist()
        to_write = []
        for row in data:
            to_write.append( [int(x) for x in row if not pd.isnull(x)] )
        with open(self.__file_dir + "/" + filename, "w") as f:
            wr = csv.writer(f, delimiter=" ")
            wr.writerows(to_write)


    def compute(self, X, input_length, num_repeats, no_details, force):
        '''
        Helper method:
        Calls SmashMatch on the specified data with the parameters specified;
            creates command string for SmashMatch binary and calls using
            SmashMatch; udpates internal variables

        Inputs -
            X (numpy.ndarray or pandas DataFrame): input data (each row is a
                different timeseries)
            input_length (int): length of the input timeseries to use
            num_repeats (int): number of times to run SmashMatch (for refining
                results)
            no_details (boolean): do not print SmashMatch processer usage and
                speed while running classification
            force (boolean): force re-classification on current dataset

        Outputs -
            (boolean) whether SmashMatch results corresponding to X were created/exist
        '''

        if force or self.should_calculate(X): # dataset was not the same as before or first run
            if force:
                self.reset_input()

            if isinstance(X, np.ndarray):
                self.__input = X
                input_name_command = " -f " + self.write_out_nda(X)
            elif isinstance(X, pd.DataFrame): # being explicit
                self.__input = X
                fname = self.get_unique_name(False)
                self.write_out_series(fname, X)
                input_name_command = " -f " + fname
            else: # theoretically should be impossible, but to be explicit
                raise ValueError("Error: unsupported types for X. X can only be of type \
                numpy.ndarray or pandas.DataFrame.")

            if input_length is not None:
                input_length_command = " -x " + str(input_length)

            self.__command += (input_name_command + self.__lib_command + "-T symbolic -D row ")
            self.__command += ("-L true true true -o " + PREFIX + " -d false")
            self.__command += (" -n " + str(num_repeats))

            if input_length is not None:
                self.__command += input_length_command
            if no_details:
                self.__command += " -t 0"

            os.chdir(self.__file_dir)
            sp.Popen(self.__command, shell=True, stderr=sp.STDOUT).wait()
            os.chdir(CWD)

            if not self.has_smashmatch(): # should theoretically be impossible \
            # to return False, but for safety
                return False
            else: # successfully ran smashmatch to get results
                return True
        else: # dataset was the same as before, use existing result files
            return True


    def reset_input(self):
        '''
        Helper Method:
        Clears the working data directory of previous run of SmashMatch and
            resets command to be fed into SmashMatch; no I/O
        '''

        os.chdir(self.__file_dir)
        sp.Popen("rm input_*", shell=True, stderr=sp.STDOUT).wait()
        sp.Popen("rm " + PREFIX + "*", shell=True, stderr=sp.STDOUT).wait()
        self.__command = self.__bin_path
        os.chdir(CWD)


    def has_smashmatch(self):
        '''
        Helper method:
        Checks internal data directory for SmashMatch result files

        Input -
            (None)
        Output -
            (boolean) True if SmashMatch files present, False if SmashMatch
                files aren't present
        '''

        if PREFIX + "_prob" in os.listdir(self.__file_dir) and \
        PREFIX + "_class" in os.listdir(self.__file_dir):
            return True
        else:
            return False


    def should_calculate(self, X_):
        '''
        Helper method:
        Determines case of current use then decides whether or not to clear
            data directory of files to prevent hogging space

        Inputs -
            X_ (numpy.ndarray or pandas.DataFrame): timeseries data

        Returns -
            True if results were cleared or interrupted and SmashMatch needs to
                be run again, False if first run or if dataset is the same
                since SmashMatch produces both the outputs of predict and predict_log_proba per every call
        '''

        # pdb.set_trace()
        # because using a np compare, have to catch self.__input = None first
        # will happen on first try
        if self.__input is None:
            return True

        if isinstance(X_, np.ndarray):
            # assuming if previous run had diff input type then now new input data
            if isinstance(self.__input, pd.DataFrame):
                self.reset_input()
                return True
            # don't clear results if same dataset: don't run again
            elif isinstance(self.__input, np.ndarray) and \
            np.array_equal(X_, self.__input):
                if has_smashmatch():
                    return False
                else: # user cancelled half-way is running ont he same input
                    return True
            # implied self.__input != X_
            elif isinstance(self.__input, np.ndarray) and \
            not np.array_equal(X_, self.__input):
                if has_smashmatch():
                    self.reset_input()
                    return True
                else: # user decided to cancel half-way and run on diff dataset
                    return True
        elif isinstance(X_, pd.DataFrame):
            if isinstance(self.__input, np.ndarray):
                self.reset_input()
                return True
            elif isinstance(self.__input, pd.DataFrame) and \
            self.__input.equals(X_):
                if self.has_smashmatch():
                    return False
                else: # re-run on the same data but user possibly cancelled half-way through
                    return True
            elif isinstance(self.__input, pd.DataFrame) and \
            not self.__input.equals(X_):
                if self.has_smashmatch():
                    self.reset_input()
                    return True
                else:
                    return True
        else: # checking cases of type
            raise ValueError("Error: unsupported types for X. X can only be of type \
            numpy.ndarray or pandas.DataFrame.")


    # actual methods of smashmatch
    def predict(self, x, il=None, nr=5, no_details=True, force=False, quantize=False):
        '''
        Classifies each input timeseries (X) using SmashMatch and the given parameters

        Inputs -
            x (numpy.nda or pandas.DataFrame): input data (each row is a
                different timeseries)
            il (int): length of the input timeseries to use
            nr (int): number of times to run SmashMatch (for refining results)
            no_details (boolean): do not print SmashMatch cpu usage/speed of
                algorithm while running clasification
            force (boolean): force re-classification on current dataset
            quantize (boolean): if input timeseries have not beeen quantized,
                apply class quantizer to input timeseries

        Outputs -
            numpy.nda of shape (num_timeseries), 1 if successful or None if not successful
        '''

        if quantize:
            assert(self.__preproc is not None), "Error: no quantization function defined"
            x = apply(self.__preproc, x)

        compute_res = self.compute(x, il, nr, no_details, force)
        if compute_res:
            class_path = self.__file_dir + "/" + PREFIX + "_class"
            with open(class_path, 'r') as f:
                raw = f.read().splitlines()
            formatted = []
            for result in raw:
                formatted.append(self.__mapper[int(result)].label) # should append labels in order
            res = np.asarray(formatted)

            # return np.reshape(res, (-1, 1))
            return res
        else:
            print("Error processing command: SmashMatch FNF. Please try again.")
            return None


    def predict_proba(self, x, il=None, nr=5, no_details=True, force=False, quantize=False):
        '''
        Predicts percentage probability for the input time series to classify as any of the possible classes fitted

        Inputs -
            x (numpy.nda or pandas.DataFrame): input data (each row is a
                different timeseries)
            il (int): length of the input timeseries to use
            nr (int): number of times to run SmashMatch (for refining results)
            no_details (boolean): do not print SmashMatch cpu usage/speed of
                algorithm while running clasification
            force (boolean): force re-classification on current dataset
            quantize (boolean): if input timeseries have not been quantized,
                apply class quantizer to input timeseries

        Outputs -
            np.nda of shape n x m if successful or None if not successful
                where n = number of timeseries and m = number of classes
                probabilities are listed in an order corresponding to the classes attribute
        '''

        if quantize:
            assert(self.__preproc is not None), "Error: no quantization function defined"
            x = apply(self.__preproc, x)

        compute_res = self.compute(x, il, nr, no_details, force)
        if compute_res:
            class_path = self.__file_dir + "/" + PREFIX + "_prob"
            probs = np.loadtxt(fname=class_path, dtype=float)
            return probs
        else:
            print("Error processing command: smashmatch FNF. Please try again.")
            return None


    def predict_log_proba(self, x, il=None, nr=5, no_details=True, force=False, quantize=False):
        '''
        Predicts logarithmic probability for the input time series to classify as any of the possible classes fitted

        Inputs -
            x (numpy.nda or pandas.DataFrame): input data (each row is a
                different timeseries)
            il (int): length of the input timeseries to use
            nr (int): number of times to run SmashMatch (for refining results)
            no_details (boolean): do not print SmashMatch cpu usage/speed of
                algorithm while running clasification
            force (boolean): force re-classification on current dataset
            quantize (boolean): if input timeseries have not been quantized,
                apply class quantizer to input timeseries

        Outputs -
            np.nda of shape n x m if successful or None if not successful
                where n = number of timeseries and m = number of classes
                probabilities are listed in an order corresponding to the classes attribute
        '''

        probs = self.predict_proba(x, il, nr, no_details, force, quantize)
        if probs is not None:
            return np.log(probs)
        else:
            return None


    def staged_fit(self, X, y, sample_weight=None, classes=None, **kwargs):
        warnings.warn('Warning: staged_fit method for this class is undefined because SmashMatch does not classify based on models.')
        pass


    def staged_predict(self, X):
        warnings.warn('Warning: staged_predict method for this class is undefined because SmashMatch does not classify based on models.')
        pass


    def staged_predict_log_proba(self, X):
        warnings.warn('Warning: staged_predict_log_proba method for this class is undefined because SmashMatch does not classify based on models.')
        pass


    def clean_libs(self):
        '''
        Helper method:
        Removes files created by reading and writing library files and clears
        relevant internally stored variables; no I/O
        '''

        os.chdir(self.__file_dir)
        sp.Popen("rm lib_*", shell=True, stderr=sp.STDOUT).wait()
        os.chdir(CWD)
        self.__classes = []
        self.__lib_files = []
        self.__lib_command = " -F "



def cleanup():
    '''
    Maintenance function:
    Delete library/result files before closing the script; no I/O
    '''

    prev_wd = os.getcwd()
    os.chdir(CWD)
    if os.path.exists(CWD + "/" + TEMP_DIR):
        command = "rm -r " + CWD + "/" + TEMP_DIR
        sp.Popen(command, shell=True, stderr=sp.STDOUT).wait()
    os.chdir(prev_wd)



atexit.register(cleanup)
