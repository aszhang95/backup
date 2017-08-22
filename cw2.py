import os, sys, time, tempfile, ntpath, csv, math, uuid, atexit
import subprocess as sp
import numpy as np



### wrapper doesn't dangerously delete files, now only dependent on correct location of where it 
### it is run relative to ./bin/smashmatch
### atexit not tested; also potentially not safe



# Global Variables:
prefix = ""  


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
        self.file_handler = tfhandler
        self.filename = path_leaf(self.file_handler.name)


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
    (and specifically bin) where tmp files can be written
    Also assumes correct modules have been loaded

    Attributes:
        classes (np.1Darray): class labels fitted into the model; also column headers for
            predict functions
    '''
    
    def __init__(self): # lib_files = list of library file names or LibFile Objects
        self.classes = [] 
        self.__lib_files = None
        self.__lib_command = " -F "
        self.__command = '../bin/smashmatch'
        self.__input_fh = None
        self.__mapper = {}
        # self.__prefix = str(uuid.uuid4())
        self.prefix = "resx"
        prefix = self.prefix


    @property
    def classes(self):
        if len(self.classes) == 0:
            print("Warning: this instance has not been fit.") # apparently can't print in a getter?
        return self.classes


    def fit(self, X, y): # not sure what to do with kwargs or the classes/sample_weight params
        '''
        Reads in appropriate data/labels -> library class files (as tempfiles) 
        to be used by smashmatch

        Inputs - 
            X (np.nda): class examples
            y (np.1da): class labels

        Returns - 
          (None) modifies object in place
        '''
        
        # clean up old library files before running
        if len(self.__lib_files) != 0:
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
            self.__lib_command += mapping[1].filename + ' '
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
            (boolean) whether the output file has been successfully created
        '''

        input_name_command = " -f " + self.read_in_nda(X)
        if input_length is not None:
            input_length_command = " -x " + str(input_length)
        if num_repeats is not None:
            num_repeats_command = " -n " + str(num_repeats)

        self.__command += (input_name_command + self.__lib_command + "-T symbolic -D row ")
        self.__command += "-L true true true -o " + self.__prefix + " -d false"

        if input_length is not None:
            self.__command += input_length_command
        if num_repeats is not None:
            self.__command += num_repeats_command

        #../bin/smashmatch  -f TEST0 -F LIB0 LIB1 LIB2 -T symbolic -D row -L true true true -o resx -n 2
        print("Requested: {}".format(self.__command))
        # sp.Popen(self.__command, shell=True).wait()
        # if prob:
        #     while not os.path.isfile(self.__prefix + "_prob"):
        #         print("Waiting for smashing algorithm to complete...")
        #         time.sleep(20)
        # else:
        #     while not os.path.isfile(self.prefix + "_class"):
        #     print("Waiting for smashing algorithm to complete...")
        #     time.sleep(20)   
        
        if (self.__prefix + "_prob" not in os.listdir(os.getcwd()) 
            or self.__prefix + "_class" not in os.listdir(os.getcwd())):
            return False
        else:
            return True


    def clear_results(self):
        '''
        Clears result files from last run of any predict method; no I/O
        '''

        if (self.__prefix + "_prob" in os.listdir(os.getcwd()) 
            or self.__prefix + "_class" in os.listdir(os.getcwd())):
            sp.Popen("rm " + self.__prefix + "*", shell=True)
            self.__lib_command = " -F "


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

        # if running predict a second time, need to clean 
        # self.clear_results()

        if self.compute(x, False, il, nr):
            with open(self.__prefix + '_class') as f:
                raw = f.read().splitlines()
            print(raw)
            os.unlink(self.__input_fh.name)
            self.__input_fh = None
            formatted = []
            for result in raw:
                formatted.append(self.__mapper[int(result)].label) # should append labels in order
            y = np.asarray(formatted)

            return np.reshape(y, (-1, 1))
        else:
            print("Error processing command. Please try again.")
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
        
        # if running predict a second time, need to clean 
        # self.clear_results()

        if self.compute(x, True, il, nr):
            probs = np.loadtxt(fname=(self.__prefix + "_prob"), dtype=float)
            os.unlink(self.__input_fh.name)
            self.__input_fh = None
            return probs
        else:
            print("Error processing command. Please try again.")
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
            print("Error processing command. Please try again.")
            return None    


    def clean_libs(self):
        '''
        Removes tempfiles created by reading and writing library files; no I/O
        '''    
        for lib_file in self.__lib_files:
            lib_file.delete_file()



def cleanup():
    '''
    Clean up library files before closing the script; no I/O
    '''
    sp.Popen("rm tmp*", shell=True).wait()
    sp.Popen("rm " + prefix + "*", shell=True).wait()



# if __name__ == '__main__':
#     launch_path = os.getcwd()
#     sp.Popen("module load gsl", shell=True).wait()
#     sp.Popen("module load boost/1.63.0+gcc-6.2", shell=True).wait()
#     sp.Popen("module load python", shell=True).wait()
#     sp.Popen("module load python", shell=True).wait()
#     # assumes you're one directory under the directory containing bin/smashmatch
#     os.chdir("../zbase", shell=true)
#     sp.Popen("make -f Makefile", shell=True).wait()
#     os.chdir("..", shell=true)
#     sp.Popen("make -f Makefile", shell=True).wait()
#     os.chdir(launch_path)
    
    
    
atexit.register(cleanup)