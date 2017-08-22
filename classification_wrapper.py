import os, sys, time, pandas
import subprocess as sp
import numpy as np

# classification Wrapper

# initiate the SMB class with the names of library files; assumes running script from same directory as 
# the lib files are contained
class SupervisedModelBase:
	def __init__(self, lib_files): # lib_files = list of library file names
		self.__lib_files = lib_files # safety precaution
		self.__lib_command = " -F "
		for file in lib_files:
			self.__lib_command += (file + " ") # have to check this syntax works
		self.__lables_dict = {}
		for i in range(len(self.__lib_files)):
			self.__labels_dict[i] = lib_files[i]

	fit(self, lib_files, sample_weight=None, classes=None):
	# what are **kwargs?
	# since data smashing is centered around not having prior knowledge about the datasets
	# traditional sklearn fit is not necessary?
	# equivalent to the setter method
		self.__lib_command = " -F "
		for file in lib_files:
			self.__lib_command += (file + " ")
		self.__labels_dict = {}
		for i in range(len(self.__lib_files)):
			self.__labels_dict[i] = lib_files[i]

	# also allow for length of input data_stream to be a parameter for predict 
	predict(self, X, num_repeats, is_nda=False): # what kind of input for X?
		'''
		Classify timeseries X using data smashing; returns most probable class
		with corresponding probability

		Inputs - 
			X (string OR nd.array if is_nda=True): collection of data streams to classify
			num_repeats (int): number of times to attempt smashmatch
			is_nda (boolean): whether input timeseries is numpy array (True) 
				or string filename (false)

		Outputs - 
			numpy array with most likely class (input library file name as first column
			and probability of that class as second column and where rows are the 
			time series)
		'''
		
		if not is_nda:
			# check to make sure you're in the same directory as the library files
			for file in self.__lib_files:
				assert file in os.listdir("."), "Error: please move to the directory \
				containing the library files. Cannot find {}.".format(file)
			command = ("../bin/smashmatch -f " + X + self.__lib_command + "-T symbolic -D row" + \
			"-L true true true -o res -n " + str(num_repeats) + " -d true")
			sp.Popen(command, shell=True).wait()
			while not os.path.isfile('resx_prob'):
				print("Waiting for data smashing to complete...")
				time.sleep(20)
			prob_dict = {}
			probs = np.loadtxt(fname="resx_prob", dtype=float)
			classes = np.loadtxt(fname="resx_class", dtype=int)
			labels = []
			for label in self.__lib_files:
				labels.append(label + "_prob")
			headers = ['nclass'] + labels


		return y

	predict_log_proba(self, X): #
		return log_probabilities_per_class


	lupi_predict(self, X):
		'''
		Perform classification on samples in the dataset X with no privileged features.
		'''

	transform(self, X):
		'''              
	 	Transform the data of standard and privileged features into standard and 
	 	generated (reconstructed) privileged features. 
	 	'''

	predict_transform(self, X):
		'''
		Transform data of only standard features to data of standard and generated 
		(reconstructed) privileged features.
		'''

	# will ask about these methods 
	# write an error message to say 
	staged_fit(self, X, y, sample_weight=None, classes=None, **kwargs):
	yield
	staged_predict(self, X):
	yield y
	staged_predict_log_proba(self, X):
	yield log_probabilities_per_class
	__getstate__(self):
	return state    
	__setstate__(self, state):
	return


# assumes starting in pycode directory but must move to where the lib_files are relative 
# to the pycode directory
if __name__ == '__main__':
	sp.Popen("module load python", shell=True).wait()
	lib_path = sys.argv[1]
	os.chdir(lib_path)