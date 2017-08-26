import os, sys, time, tempfile, ntpath, csv, math, uuid, atexit
import subprocess as sp
import numpy as np



### wrapper doesn't dangerously delete files, now only dependent on correct location of where it 
### it is run relative to ./bin/smashmatch



# Global Variables: (ensures safe R/W of smashmatch)
global prefix
prefix = str(uuid.uuid4())
# prefix = "resx" # for testing purposes only
libs = ['LIB0', 'LIB1', 'LIB2']
labs = [0, 1, 2]


# necessary helper global function
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
        filename (string): name of the temporary file within the cwd of a library file
    '''
    
    def __init__(self, class_, label_, tfhandler):
        self.class_name = class_
        self.label = label_
        self.file_handler = tfhandler # can be string or tempfile obj
        if not isinstance(tfhandler, basestring): 
            self.filename = path_leaf(self.file_handler.name)
        else:
            self.filename = tfhandler



    @property
    def class_name(self):
        return self.class_name


    @property
    def label(self):
        return self.label


    @property
    def file_handler(self):
        return self.file_handler


    @property
    def file_name(self):
        return self.filename


    @class_name.setter
    def class_name(self, class_):
        self.class_name = class_


    @label.setter
    def label(self, label_):
        self.label = label_


    @file_handler.setter
    def file_handler(self, tfhandler):
        self.file_handler = tfhandler


    def delete_file(self):
        '''
        Deletes temporary library file created from fit function; no I/O
        '''
        os.unlink(self.file_handler.name)



class SupervisedModelBase:
    '''
    Object for smashmatch classification; modeled after sklearn.SVM classifier and using 
    D3M API specifications
    
    ### USAGE ###
    Assumes you are within directory one level lower than data_smashing_ 
    (and specifically bin) where temporary files can be written

    Attributes:
        classes (np.1Darray): class labels fitted into the model; also column headers for
            predict functions
    '''
    
    def __init__(self): # lib_files = list of library file names or LibFile Objects
        self.classes = [] 
        self.__lib_files = []
        self.lib_command = " -F "
        self.command = '../bin/smashmatch'
        self.__input_fh = None
        self.input = None
        self.__mapper = {}


    def fit_files(self, filenames, labels):
        '''
        Use when input data is not of type np.array; assumes indices of both lists correlate

        Inputs - 
            filenames (list of strings)
            labels (list of ints)

        Outputs -
            (None) modifies object in place
        '''
        
        # clear previous library files if necessary        
        len_libs = len(self.__lib_files)
        if len_libs != 0:
            self.clean_libs()

        # mapper is key=class_num, value=LibFile
        for i in range(len(labels)):
            lf = LibFile(i, labels[i], filenames[i])
            self.__mapper[i] = lf # key=class_num, value=LibFile
            self.lib_command += filenames[i] + ' '
            self.classes.append(labels[i])
        self.classes = np.asarray(self.classes)


    def fit(self, X, y): # not sure what to do with kwargs or the classes/sample_weight params
        '''
        Reads in appropriate data/labels -> library class files (as tempfiles) 
        to be used by smashmatch

        Inputs - 
            X (np.nda): class examples
            y (np.1da): class labels

        Outputs - 
          (None) modifies object in place
        '''
        
        # delete old library files before running (would only be true after first run)
        len_libs = len(self.__lib_files)
        if len_libs > 0:
            self.clean_libs()

        self.__lib_files = self.make_libs(X, y)
        mappings = []
        # need to make sure we feed the class_names in according to their actual order
        for class_ in self.__lib_files:
            mappings.append((class_.class_name, class_))
        mappings.sort(key=lambda x: x[0])
        for mapping in mappings:
            self.__mapper[mapping[0]] = mapping[1] # key = class_num, value = LibFile
            self.classes.append(mapping[1].label)
            self.lib_command += mapping[1].filename + ' '
        self.classes = np.asarray(self.classes)
        # return self
        

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
            fh = tempfile.NamedTemporaryFile(dir=os.getcwd(), delete=False)
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
        self.__input_fh = tempfile.NamedTemporaryFile(dir=os.getcwd(), delete=False)    
        wr = csv.writer(self.__input_fh, delimiter=" ")
        wr.writerows(rows)
        self.__input_fh.close()
        return path_leaf(self.__input_fh.name)


    def compute(self, X, prob, input_length, num_repeats):
        '''
        Helper to call smashmatch on the specified input file with the parameters specified

        Inputs - 
            X (nda): input data (each row is a different timeseries)
            input_length (int): length of the input timeseries to use 
            num_repeats (int): number of times to run smashmatch (for refining results)
            prob (boolean): to interface with predict class or predict probability
                when prob is True compute looks for output probabilities
                when prob is False compute looks for output classes
        Outputs - 
            (boolean) whether smashmatch results corresponding to X were created/exist
        '''

        ran_smashmatch = False
        
        if isinstance(X, str) and self.should_calculate_file(X):
            input_name_command = " -f " + X
            run_smashmatch = True
        elif self.should_calculate(X): # dataset was not the same as before or first run
            input_name_command = " -f " + self.read_in_nda(X)
            run_smashmatch = True

        if run_smashmatch:
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
            if prob:
                while not os.path.isfile(prefix + "_prob"):
                    print("Waiting for smashing algorithm to complete...")
                    time.sleep(20)
            else:
                while not os.path.isfile(prefix + "_class"):
                    print("Waiting for smashing algorithm to complete...")
                    time.sleep(20)   
            
            if prefix + "_prob" not in os.listdir(os.getcwd()) or \
            prefix + "_class" not in os.listdir(os.getcwd()):
                return False
            else: # successfully ran smashmatch to get results
                os.unlink(self.__input_fh.name)
                self.__input_fh = None
                return True
        else: # dataset was the same as before, use existing result files
            return True


    def should_calculate_file(self, X_):
        '''
        Determines logistics of reading in from file as opposed to np.nda in similar
        fashion to should_calculate(np.nda)

        Inputs -
            X_ (string filename)

        Returns -
            boolean 
        '''

        if self.input is None:
            return True
        elif isinstance(self.__input, str) and self.input != X_:
            sp.Popen("rm " + prefix + "*", shell=True).wait()
            self.command = '../bin/smashmatch'
            return True
        elif isinstance(self.__input, str) and self.input == X_:
            return False
        elif not isinstance(self.__input, str):
            sp.Popen("rm " + prefix + "*", shell=True).wait()
            self.command = '../bin/smashmatch'
            return True
        else: # need to test logic 
            print("Not first run and no smashmatch files could be found! Retrying...")
            self.command = '../bin/smashmatch'
            return True


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
        # don't clear results if same dataset: don't run again
        elif np.array_equal(X_, self.input): 
            return False
        elif prefix + "_prob" in os.listdir(os.getcwd()) and \
        prefix + "_class" in os.listdir(os.getcwd()): # implied self.input != X_
            sp.Popen("rm " + prefix + "*", shell=True).wait()
            self.command = '../bin/smashmatch'
            return True
        else: # should only be one of the 3 above cases, but want to be explicit
        # surprise could happen if only either prefix_prob or prefix_class exist
        # logic should be tested
            sys.exit(1)


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

        compute_res = self.compute(x, False, il, nr)
        if compute_res and prefix + "_class" in os.listdir(os.getcwd()):
            with open(prefix + '_class') as f:
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

        compute_res = self.compute(x, True, il, nr)
        if compute_res and prefix + "_prob" in os.listdir(os.getcwd()):
            probs = np.loadtxt(fname=(prefix + "_prob"), dtype=float)
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
        lib_type = self.__lib_files[i].file_handler
        if not isinstance(lib_type, str):
            for lib_file in self.__lib_files:
                lib_file.delete_file()
        self.classes = [] 
        self.__lib_files = []
        self.lib_command = " -F "



def cleanup():
    '''
    Clean up library files before closing the script; no I/O
    '''
    
    sp.Popen("rm tmp*", shell=True).wait()
    if prefix != "" and prefix is not None:
        sp.Popen("rm " + prefix + "*", shell=True).wait()



# if __name__ == '__main__':
#     launch_path = os.getcwd()
#     sp.Popen("module load gsl", shell=True).wait()
#     sp.Popen("module load boost/1.63.0+gcc-6.2", shell=True).wait()
#     sp.Popen("module load python", shell=True).wait()
#     sp.Popen("module load python", shell=True).wait()
#     # assumes you're one directory under the directory containing bin/smashmatch
#     os.chdir("../zbase")
#     sp.Popen("make -f Makefile", shell=True).wait()
#     os.chdir("..")
#     sp.Popen("make -f Makefile", shell=True).wait()
#     os.chdir(launch_path)
    
    
    
atexit.register(cleanup)