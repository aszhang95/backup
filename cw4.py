import os, sys, time, tempfile, ntpath, csv, math, uuid, atexit
import subprocess as sp
import numpy as np
from numpy import nan
import pandas as pd



# Global Variables: (ensures safe R/W of smashmatch)
prefix = str(uuid.uuid4())
temp_dir = str(uuid.uuid4())
cwd = os.getcwd()
# prefix = "resx" # for testing purposes only


# necessary global function
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



class LibFile:
    '''
    Helper class to store library file information to interface with file-I/O data smashing code

    Attributes -
        class (int):  label given to the class by smashmatch based on the order in
            which it was processed
        label (int): actual label assigned by input
        file_handler (tempfile.NamedTemporaryFile): temporary file object reference for R/W
            interfacing with smashmatch
        filename (string): path to temporary library file
    '''

    def __init__(self, class_, label_, tfhandler):
        self.class_name = class_
        self.label = label_
        self.file_handler = tfhandler
        self.filename = self.file_handler.name


    def delete_file(self):
        '''
        Deletes temporary library file created from fit function; no I/O
        '''
        os.unlink(self.file_handler.name)



class SupervisedModelBase:
    '''
    Object for smashmatch classification; modeled after sklearn.SVM classifier and using
    D3M API specifications

    Inputs -
        bin_path_(string): Path to smashmatch as a string

    Attributes:
        classes (np.1Darray): class labels fitted into the model; also column headers for
            predict functions
    '''

    def __init__(self, bin_path_): # lib_files = list of library file names or LibFile Objects
        self.bin_path = bin_path_
        sp.Popen("mkdir "+ temp_dir, shell=True).wait()
        self.file_dir = cwd + "/" + temp_dir
        # os.chdir(self.file_dir) # move into the directory for safer functionality or stay in cwd?
        self.classes = []
        self.__lib_files = []
        self.lib_command = " -F "
        self.command = self.bin_path
        self.input_fh = None
        self.input = None
        self.__mapper = {}


    ### helper functions for library files:
    def read_in_ragged(self, filename, delimiter_, datatype=float):
        '''
        Reads in file with mixed column lengths (timeseries of different length)

        Inputs -
            file (string): path to file
            delimiter (char): value delimiter
            nrows (int): number of timeseries (optional, but would speed up process)
            max_cols (int): length of max (optional, but would speed up process)
        Outputs -
            pandas.DataFrame with missing values as NaN
        '''

        data = []
        max_col_len = 0
        with open(filename, 'r') as f:
            for line in f:
                row = line.rstrip('\n').split(delimiter_)
                row_len = len(row)
                if row_len > max_col_len:
                    max_col_len = row_len
                data.append(row)
        rv = pd.DataFrame(data, columns=range(max_col_len), dtype=datatype)
        rv.fillna(value=nan, inplace=True)
        return rv


    def write_out_ragged(self, df_):
        '''
        Writes out pd.DataFrame to tempfile

        Inputs -
            df (pd.DataFrame)

        Outputs -
            out (tempfile)
        '''
        assert(isinstance(df_, pd.DataFrame)), "Error: argument df is not type pandas.DataFrame"

        data = df_.values.tolist()
        to_write = []
        for row in data:
            to_write.append([x for x in row if not pd.isnull(x)])
        out = tempfile.NamedTemporaryFile(dir=self.file_dir, delete=False)
        wr = csv.writer(out, delimiter=" ")
        wr.writerows(to_write)
        out.close()
        return out


    def condense(self, mappings):
        '''
        Creates X, y necessary for smashmatch following sklearn.SVM conventions for the fit method

        Input -
            mappings(list of tuples of (df, label))
        Output -
            X (examples of each class, each example is a row)
            y (df with corresponding of n x 1 with n being the number of timeseries in X)
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
        Helper function to write class labels & examples to files usable by smashmatch,
        but also compatible with pandas DataFrames

        Inputs -
            X (pd.DataFrame): timeseries examples of each class
            y (pd.Dataframe): labels for each type of timeseries
        Returns -
            lib_files (list of LibFile objects)
        '''

        X.insert(0, "level", y)
        X.sort_values("level")

        labels = y.unique().tolist()
        class_num = 0

        lib_files = []
        for label_ in labels:
            df = X.loc[X.level == label_]
            df.drop("level", 1)
            fh = self.write_out_ragged(df)
            lib_files.append(LibFile(class_num, label_, fh))
            class_num += 1
        return lib_files


    def make_libs(self, X, y):
        '''
        Helper function to write class labels & examples to files usable by smashmatch

        Inputs -
            X (np.nda): class examples
            y (np.1da): class labels

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
            fh = tempfile.NamedTemporaryFile(dir=self.file_dir, delete=False)
            lib_file = LibFile(class_num, label_, fh)
            rows = []
            for stream in class_:
                rows.append(stream[1:].tolist()) # cut off label from row entry
            wr = csv.writer(fh, delimiter=" ")
            wr.writerows(rows)
            fh.close()
            class_num += 1
            rv.append(lib_file)
        return rv


    def fit(self, X, y): # not sure what to do with kwargs or the classes/sample_weight params
        '''
        Reads in appropriate data/labels -> library class files (as tempfiles)
        to be used by smashmatch

        Inputs -
            X (np.nda or pandas.DataFrame): class examples
            y (np.1da or pandas.Series): class labels

        Returns -
          (None) modifies object in place
        '''

        # delete old library files before running (would only be true after first run)
        len_libs = len(self.__lib_files)
        if len_libs != 0:
            self.clean_libs()

        if isinstance(X, np.ndarray):
            self.__lib_files = self.make_libs(X, y)
        elif isinstance(X, pd.DataFrame):
            self.__lib_files = self.make_libs_df(X, y)
        else:
            raise ValueError("Error: unsupported types for X. X can only be of type \
            numpy.ndarray or pandas.DataFrame.")

        mappings = []
        # need to make sure we feed the class_names in according to their actual order
        for lib in self.__lib_files:
            mappings.append((lib.class_name, lib))
        mappings.sort(key=lambda x: x[0])
        for mapping in mappings:
            self.__mapper[mapping[0]] = mapping[1] # key = class_num, value = LibFile
            self.classes.append(mapping[1].label)
            self.lib_command += mapping[1].filename + ' ' # should return full path to file
        self.classes = np.asarray(self.classes)
        # return self


    def read_in_nda(self, array):
        '''
        Helper function to read in input data to file usable by smashmatch

        Inputs -
            array (np.nda): data to be classified; each row represents a different timeseries
                to be classified
        Outputs -
            (string) filename of the input_file as a tempfile
        '''

        self.input = array
        rows = array.tolist()
        self.input_fh = tempfile.NamedTemporaryFile(dir=os.getcwd(), delete=False)
        wr = csv.writer(self.input_fh, delimiter=" ")
        wr.writerows(rows)
        self.input_fh.close()
        return self.input_fh.name


    def compute(self, X, input_length, num_repeats):
        '''
        Helper to call smashmatch on the specified input file with the parameters specified

        Inputs -
            X (nda): input data (each row is a different timeseries)
            input_length (int): length of the input timeseries to use
            num_repeats (int): number of times to run smashmatch (for refining results)
        Outputs -
            (boolean) whether smashmatch results corresponding to X were created/exist
        '''

        if self.should_calculate(X): # dataset was not the same as before or first run
            if isinstance(X, np.ndarray):
                input_name_command = " -f " + self.read_in_nda(X)
            elif isinstance(X, pd.DataFrame): # being explicit
                input_name_command = " -f " + self.write_out_ragged(X).name
            else:
                raise ValueError("Error: unsupported types for X. X can only be of type \
                numpy.ndarray or pandas.DataFrame.")

            if input_length is not None:
                input_length_command = " -x " + str(input_length)
            if num_repeats is not None:
                num_repeats_command = " -n " + str(num_repeats)

            self.command += (input_name_command + self.lib_command + "-T symbolic -D row ")
            self.command += "-L true true true -o " + prefix + " -d false"

            if input_length is not None:
                self.command += input_length_command
            if num_repeats is not None:
                self.command += num_repeats_command

            # (../bin/smashmatch  -f TEST0 -F LIB0 LIB1 LIB2
            # -T symbolic -D row -L true true true -o resx -n 2)
            print("Requested: {}".format(self.command))
            sp.Popen(self.command, shell=True).wait()

            while not self.has_smashmatch():
                print("Waiting for smashing algorithm to complete...")
                time.sleep(20)

            if not self.has_smashmatch(): # should theoretically be impossible \
            # to return False, but for safety
                return False
            else: # successfully ran smashmatch to get results
                os.unlink(self.input_fh.name)
                self.input_fh = None
                return True
        else: # dataset was the same as before, use existing result files
            return True


    def reset_input(self):
        '''
        Clears the working data directory of previous run of smashmatch; no I/O
        '''

        os.chdir(self.file_dir)
        sp.Popen("rm " + prefix + "*", shell=True).wait()
        self.command = self.bin_path
        os.chdir(cwd)


    def has_smashmatch(self):
        '''
        Checks data directory for smashmatch files

        Input -
            (None)
        Output -
            (boolean) True if smashmatch files present, False if smashmatch files aren't present
        '''

        if prefix + "_prob" in os.listdir(self.file_dir) and \
        prefix + "_class" in os.listdir(self.file_dir):
            return True
        else:
            return False


    def should_calculate(self, X_):
        '''
        Clears result files of smashmatch if the previous dataset is different than the current
        or if this is the first run of smashmatch (in which case self.input would be None)

        Inputs -
            X_ (nda): input time series

        Returns -
            True if results were cleared and smashmatch needs to be run again, False if
                first run or if dataset is the same
            Or will exit abruptly if unexpected 4 case arises
        '''

        # because using a np compare, have to catch self.input = None first
        # will happen on first try
        if self.input is None:
            return True

        if isinstance(X_, np.ndarray):
            # assuming if previous run had diff input type then now new input data
            if isinstance(self.input, pd.DataFrame):
                self.reset_input()
                return True
            # don't clear results if same dataset: don't run again
            elif isinstance(self.input, np.ndarray) and \
            np.array_equal(X_, self.input) and self.has_smashmatch():
                return False
            # implied self.input != X_
            elif isinstance(self.input, np.ndarray) and not np.array_equal(X_, self.input):
                self.reset_input()
                return True
            else: # should only be one of the 3 above cases, but want to be explicit
            # surprise could happen if only either prefix_prob or prefix_class exist
                # recalculating seems like best option here if the files don't exist
                print("Entered surprise state! Spoopy.")
                self.command = self.bin_path
                return True
        elif isinstance(X_, pd.DataFrame):
            if isinstance(self.input, np.ndarray):
                self.reset_input()
                return True
            elif isinstance(self.input, pd.DataFrame) and self.input.equals(X_) and \
            self.has_smashmatch():
                return False
            elif isinstance(self.input, pd.DataFrame) and not self.input.equals(X_):
                self.reset_input()
                return True
            else:
                print("Entered surprise state! Spoopy.")
                self.command = self.bin_path
                return True
        else:
            raise ValueError("Error: unsupported types for X. X can only be of type \
            numpy.ndarray or pandas.DataFrame.")


    def predict(self, x, il=None, nr=None):
        '''
        Classifies each of the input time series (X) using smashmatch and the given parameters

        Inputs -
            X (nda): input data (each row is a different timeseries)
            il (int): length of the input timeseries to use (smashmatch param)
            nr (int): number of times to run smashmatch (for refining results) (smashmatch param)

        Outputs -
            np.nda of shape num_timeseries, 1 if successful or None if not successful
        '''

        compute_res = self.compute(x, il, nr)
        if compute_res:
            class_path = self.file_dir + "/" + prefix + "_class"
            with open(class_path, 'r') as f:
                raw = f.read().splitlines()
            formatted = []
            for result in raw:
                formatted.append(self.__mapper[int(result)].label) # should append labels in order
            y = np.asarray(formatted)

            return np.reshape(y, (-1, 1))
        else:
            print("Error processing command: smashmatch FNF. Please try again.")
            return None


    def predict_proba(self, x, il=None, nr=None):
        '''
        Predicts percentage probability for the input time series to classify as any
        of the possible classes fitted

        Inputs -
            X (nda): input data (each row is a different timeseries)
            il (int): length of the input timeseries to use (smashmatch param)
            nr (int): number of times to run smashmatch (for refining results) (smashmatch param)

        Outputs -
            np.nda of shape n x m if successful or None if not successful
                where n = num_timeseries and m = num_classes
                probabilities are listed in an order corresponding to the classes attribute
        '''

        compute_res = self.compute(x, il, nr)
        if compute_res:
            class_path = self.file_dir + "/" + prefix + "_prob"
            probs = np.loadtxt(fname=class_path, dtype=float)
            return probs
        else:
            print("Error processing command: smashmatch FNF. Please try again.")
            return None


    def predict_log_proba(self, x, il=None, nr=None):
        '''
        Predicts logarithmic probability for the input time series to classify as any
        of the possible classes fitted

        Inputs -
            X (nda): input data (each row is a different timeseries)
            input_length (int): length of the input timeseries to use
            num_repeats (int): number of times to run smashmatch (for refining results)

        Outputs -
            np.nda of shape n x m if successful or None if not successful
                where n = num_timeseries and m = num_classes
                probabilities are listed in an order corresponding to the classes attribute
        '''

        probs = self.predict_proba(x, il, nr)
        if probs is not None:
            return np.log(probs)
        else:
            return None


    def clean_libs(self):
        '''
        Removes tempfiles created by reading and writing library files and clears
        relevant internally stored variables; no I/O
        '''

        for lib_file in self.__lib_files:
            lib_file.delete_file()
        self.classes = []
        self.__lib_files = []
        self.lib_command = " -F "



def cleanup():
    '''
    Clean up library files before closing the script; no I/O
    '''

    os.chdir(cwd)
    if os.path.exists(cwd + "/" + temp_dir):
        command = "rm -r " + cwd + "/" + temp_dir
        sp.Popen(command, shell=True).wait()

    # sp.Popen("rm tmp*", shell=True).wait()
    # if prefix != "" and prefix is not None:
    #     sp.Popen("rm " + prefix + "*", shell=True).wait()


atexit.register(cleanup)
