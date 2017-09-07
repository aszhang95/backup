"""
Unsupervised Learning API for Series Modeling for D3M

Three classes are defined:

+ Input
+ Output (using the Output class is optional, but will probably help TA2)
+ Unsupervised_Series_Learning_Base (abstract base class)

"""

__version__ = "v0.31415"


from abc import ABCMeta, abstractmethod
import numpy as np
import warnings

class Input(object):
    """
    # Class for data input ([time] series data corpora)

    + data :type: numpy.ndarray, dtype=int|double|bool|object
    + is_synchronized :type: bool
    + is_categorical :type: bool
    + is_multiple_streams :type: bool
    + preproc :type: function: numpy.ndarray+parameters -> numpy.ndarray
    + get : type :function: void -> numpy.ndarray

    data is a numpy.ndarray with ndim<=2. Axis 0 is the sequential
    data (or time) and (rows) axis 1 are the different streams.

    dtype for the ndarray must be specified as dtype=object
    if the values are not double, or integer or bool.

    If dtype is specified as object, then property is_categorical=True
    always. If data.dtype is np.dtype(np.bool), then one may
    explicitly set is_categorical=True. Otherwise we raise error
    (e.g. if is_categorical=True and data.dtype is np.dtype(np.float64))

    In the multi-stream case, series data of unequal length is padded
    on right by np.NaN.

    Property is_synchronized=True implies that the data streams
    are synchronized in time (and hence not independent),
    in which case np.Nan may occur not just as right padding in order
    to signal missing data in such synchronized streams.

    Property force_vect_preproc can be set to true if input preproc function
    is not already vectorized (necessary for mapping over unquantized data)

    Function preproc is used for modifying the input data
    (returned from get()). The expected use case is quantization,
    but can be used for general pre-processing.

    Function get returns preproced data

    """
    def __init__(self,data=np.empty([1,1]),
                 is_categorical=False,
                 is_synchronized=False,
                 preproc=None, force_vect_preproc=True):
        self._data=data
        self.is_categorical=is_categorical
        self.check_data()
        self.is_synchronized=is_synchronized
        if preproc is not None:
            if force_vect_preproc:
                self._preproc=np.vectorize(preproc)
            else:
                self._preproc=preproc
            self.check_preproc(self._preproc)


    @property
    def data(self):
        return self._data

    @data.setter
    def data(self,input_data):
        self._data = input_data
        self.check_data()

    @property
    def preproc(self):
        return self._preproc

    @preproc.setter
    def preproc(self,func,force_vect_preproc=False):
        if force_vect_preproc:
            self._preproc = np.vectorize(func)
        else:
            self._preproc=func
        self.check_preproc(self._preproc)


    def get(self):
        if self._preproc is not None:
            return self._preproc(self._data)
        else:
            print("Error: no preproc function defined")
            return None

    def _preproc(d):
        return d

    def check_preproc(self,func):
        if not callable(func):
            raise Exception('preproc must be ndarray -> ndarray')
        if (type(func(np.ones([1,3]))) is not np.ndarray):
            raise Exception('dtype for preproc must be ndarray -> ndarray')
        return True

    def check_data(self):
        if((self._data.dtype  is np.int64)
           |(self._data.dtype is np.float64)
           |(self._data.dtype is np.bool)
           |(self._data.dtype is object)):
            raise Exception('dtype for input _data int|bool|float|object')

        if((self._data.dtype  is np.dtype(np.int64))
           |(self._data.dtype is np.dtype(np.float64))):
            self.is_categorical=False

        if self._data.ndim > 1:
            self.is_multiple_streams = True
        else:
            self.is_multiple_streams = False

    def __str__(self):
        return "data: %s \nis_categorical : %s \nis_multistream : %s\nis_synchronized : %s" % \
            (self._data,
             self.is_categorical,
             self.is_multiple_streams,
             self.is_synchronized)
    #-----------------------------




class Unsupervised_Series_Learning_Base(object):
    """
    # Unsupervised Series Learning Abstract Base class

    ## Attributes:

    + _problem_type : type : str

    Specifies a string description of problem type
    e.g. clustering, regression etc. _PROBLEMS_ specifies a list of expected problem types.
    Other strings produce a warning.

    + _data : type : Input

    + _data_y : type : Input (in case we want to
    have some kind of auxillary data), e.g. labels of series data

    + _hyperparameters : type : dict { str : value }

    + _primitive : type : path_to_primitive


    ## Methods:

    These must be implemented in derived class, and would
    otherwise raise error:

    + fit :  standard interpretation as in sklearn
    + fit_transform :  standard interpretation as in sklearn
    + predict :  standard interpretation as in sklearn
    + transform :  standard interpretation as in sklearn
    + fit_preditc :  standard interpretation as in sklearn
    + predict_proba :  standard interpretation as in sklearn
    + predict_log_proba :  standard interpretation as in sklearn

    + performance :  any appropriate performance measure (optional)
    + score :  standard interpretation as in sklearn

    + set_param  : standard interpretation as in sklearn
    + get_param  : standard interpretation as in sklearn

    """
    __metaclass__ = ABCMeta

    _PROBLEMS_=set(['clustering','distance_metric_learning',
                    'featurization','embedding','regression'])

    def __init__(self,problem_="clustering"):
        self._output = None
        self._problem_type=problem_

    @property
    def output(self):
        return self._output

    @property
    def data(self):
        return self._data

    @property
    def data_y(self):
        return self._data_additional

    @property
    def hyperparameters(self):
        return self._hyperparameters

    @property
    def primitive(self):
        return self._primitive

    @property
    def problem_type(self):
        return self._problem_type


    @data.setter
    def data(self,input_data):
        if not isinstance(input_data,Input):
            raise Exception('data must be instance of Input class')
        self._data=input_data


    @data_y.setter
    def data_y(self,input_data):
        if not isinstance(input_data,Input):
            raise Exception('additional data must be instance of Input class')
        self._data_additional=input_data

    @hyperparameters.setter
    def hyperparameters(self,dict_):
        if self.hyper_parameter_check(dict_):
            self._hyperparameters=dict_


    @primitive.setter
    def primitive(self,prim_):
        if self.primitive_check(prim_):
            self._primitive=prim_


    @problem_type.setter
    def problem_type(self,flag):
        if type(flag) is not str:
            raise Exception('Problem_type expects str')
        if flag  not in _PROBLEMS_:
            warnings.warn('Warning: Uncharted learning problem')
        self._problem_type=flag


    def hyper_parameter_check(self,dict_):
        if not isinstance(dict_,dict):
            raise Exception('hyperparametrs need to specified as a dict,\
            or a dict-like interface, with name : value pairs')
        for key,value in dict_.iteritems():
            if type(key) is not str:
                raise Exception('hyperparameter name expects str')
        return True

    # suggest defining in inherited class
    def primitive_check(dict_):
        return True


    @abstractmethod
    def fit(self,*arg,**kwds):
        pass

    @abstractmethod
    def fit_transform(self,*arg,**kwds):
        pass

    @abstractmethod
    def fit_predict(self,*arg,**kwds):
        pass

    @abstractmethod
    def predict(self,*arg,**kwds):
        pass

    @abstractmethod
    def predict_proba(self,*arg,**kwds):
        pass

    @abstractmethod
    def log_proba(self,*arg,**kwds):
        pass

    @abstractmethod
    def score(self,*arg,**kwds):
        pass

    @abstractmethod
    def transform(self,*arg,**kwds):
        pass

    def set_param(self,**kwargs):
        self._hyperparameters=kwargs
        return

    def get_param(self):
        return self._hyperparameters


#EOF
