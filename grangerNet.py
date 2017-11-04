from abc import ABCMeta, abstractmethod
import warnings


class spatiotemporal_data_(object):
    """class for reading in spatio temporal data, and provide it in a
    format easily ingestable by the XgenESeSS binaries.

    Targeted funtionality:

        1. Read in data assumed to be in the following tabular format:

        <data_location> <timestamp> <data_attribute>
        <data_location> <timestamp> <data_attribute>
             ...           ...            ...

        where a) the field ordering might change (and should be specifiable),
              b) the time stamp format must be flexible,
              c) and can be multiple attributes.

        2. Return metadata about data. Some of which might be results of
           simple statistical computations. Return in dictionary
           accessed by method @meta_properties@

        3. Spatio-temporal quantization method  @transform@. Returns ndarray, indexmap

        4. Pull data from web interface specified in livepath by @pull@

        5. Legacy support for existing data directory.

    Attributes:
        _indexmap: map from data index to location
        _path:     path to data log
        _livepath: webpath to data interface
        _meta_dict: dict of meta properties

    """

    __metaclass__ = ABCMeta

    def __init__(self,path,
                 livepath=None):
        self._path=path
        self._livepath=livepath
        self._indexmap=None

    @property
    def indexmap(self):
        return self._indexmap

    @abstractmethod
    def pull(self, *arg, **kwds):
        """pull data from webpage

        Args:
            path

        Returns:
            log file and/or self object

        Raises:
            When errors or inaccessible
        """
        pass

    @abstractmethod
    def meta_properties(self, *arg, **kwds):
        """compute meta properties from data
        which include: size and complexity of data set,
        location properties, time span, number of attributes if any

        Args:
            keyword specifying wht property to compute

        Returns:
            A dict mapping keywords to properties

        Raises:
            Unimplemeneted keyword. Might be a warning
        """
        pass

    @abstractmethod
    def transform(self, *arg, **kwds):
        """transforms tabular data to quantized time series corpora,
        given the spatial and temporal quantization parameters


        Args:
            data: dataframe?
            spatial quantization parameters
            temporal quantization parameters

        Returns:
            quantized time series corpora,
            indexmap

        Raises:
            When quantization specified is impossible
        """
        pass



class zed_binary_(object):
    """Summary of class here.

    Longer class information....
    Longer class information....

    Attributes:
        likes_spam: A boolean indicating if we like SPAM or not.
        eggs: An integer count of the eggs we have laid.
    """

    def __init__(self,dict_):
        self._name_path=dict_



class cynet_(object):
    """Summary of class here.

    Longer class information....
    Longer class information....

    Attributes:
        likes_spam: A boolean indicating if we like SPAM or not.
        eggs: An integer count of the eggs we have laid.
    """

    __metaclass__ = ABCMeta

    def __init__(self,modeldir,data,cynet_binary):
        self._modeldir=modeldir
        self._data=data
        self._cynet_binary=cynet_binary


    @abstractmethod
    def fit(self, *arg, **kwds):
        """Set up causality network with models. Set up data access

        Retrieves rows pertaining to the given keys from the Table instance
        represented by big_table.  Silly things may happen if
        other_silly_variable is not None.

        Args:
            big_table: An open Bigtable Table instance.
            keys: A sequence of strings representing the key of each table row
                to fetch.
            other_silly_variable: Another optional variable, that has a much
                longer name than the other args, and which does nothing.

        Returns:
            A dict mapping keys to the corresponding table row data
            fetched. Each row is represented as a tuple of strings. For
            example:

            {'Serak': ('Rigel VII', 'Preparer'),
             'Zim': ('Irk', 'Invader'),
             'Lrrr': ('Omicron Persei 8', 'Emperor')}

            If a key from the keys argument is missing from the dictionary,
            then that row was not found in the table.

        Raises:
            IOError: An error occurred accessing the bigtable.Table object.
        """
        pass

    @abstractmethod
    def predict(self, *arg, **kwds):
        """Fetches rows from a Bigtable.

        Retrieves rows pertaining to the given keys from the Table instance
        represented by big_table.  Silly things may happen if
        other_silly_variable is not None.

        Args:
            big_table: An open Bigtable Table instance.
            keys: A sequence of strings representing the key of each table row
                to fetch.
            other_silly_variable: Another optional variable, that has a much
                longer name than the other args, and which does nothing.

        Returns:
            A dict mapping keys to the corresponding table row data
            fetched. Each row is represented as a tuple of strings. For
            example:

            {'Serak': ('Rigel VII', 'Preparer'),
             'Zim': ('Irk', 'Invader'),
             'Lrrr': ('Omicron Persei 8', 'Emperor')}

            If a key from the keys argument is missing from the dictionary,
            then that row was not found in the table.

        Raises:
            IOError: An error occurred accessing the bigtable.Table object.
        """
        pass

    @abstractmethod
    def generate_table(self, *arg, **kwds):
        """Fetches rows from a Bigtable.

        Retrieves rows pertaining to the given keys from the Table instance
        represented by big_table.  Silly things may happen if
        other_silly_variable is not None.

        Args:
            big_table: An open Bigtable Table instance.
            keys: A sequence of strings representing the key of each table row
                to fetch.
            other_silly_variable: Another optional variable, that has a much
                longer name than the other args, and which does nothing.

        Returns:
            A dict mapping keys to the corresponding table row data
            fetched. Each row is represented as a tuple of strings. For
            example:

            {'Serak': ('Rigel VII', 'Preparer'),
             'Zim': ('Irk', 'Invader'),
             'Lrrr': ('Omicron Persei 8', 'Emperor')}

            If a key from the keys argument is missing from the dictionary,
            then that row was not found in the table.

        Raises:
            IOError: An error occurred accessing the bigtable.Table object.
        """
        pass

    @abstractmethod
    def compute_property(self, *arg, **kwds):
        """Fetches rows from a Bigtable.

        Retrieves rows pertaining to the given keys from the Table instance
        represented by big_table.  Silly things may happen if
        other_silly_variable is not None.

        Args:
            big_table: An open Bigtable Table instance.
            keys: A sequence of strings representing the key of each table row
                to fetch.
            other_silly_variable: Another optional variable, that has a much
                longer name than the other args, and which does nothing.

        Returns:
            A dict mapping keys to the corresponding table row data
            fetched. Each row is represented as a tuple of strings. For
            example:

            {'Serak': ('Rigel VII', 'Preparer'),
             'Zim': ('Irk', 'Invader'),
             'Lrrr': ('Omicron Persei 8', 'Emperor')}

            If a key from the keys argument is missing from the dictionary,
            then that row was not found in the table.

        Raises:
            IOError: An error occurred accessing the bigtable.Table object.
        """
        pass

    @abstractmethod
    def evaluate_variable_association(self, *arg, **kwds):
        """Fetches rows from a Bigtable.

        Retrieves rows pertaining to the given keys from the Table instance
        represented by big_table.  Silly things may happen if
        other_silly_variable is not None.

        Args:
            big_table: An open Bigtable Table instance.
            keys: A sequence of strings representing the key of each table row
                to fetch.
            other_silly_variable: Another optional variable, that has a much
                longer name than the other args, and which does nothing.

        Returns:
            A dict mapping keys to the corresponding table row data
            fetched. Each row is represented as a tuple of strings. For
            example:

            {'Serak': ('Rigel VII', 'Preparer'),
             'Zim': ('Irk', 'Invader'),
             'Lrrr': ('Omicron Persei 8', 'Emperor')}

            If a key from the keys argument is missing from the dictionary,
            then that row was not found in the table.

        Raises:
            IOError: An error occurred accessing the bigtable.Table object.
        """
        pass
