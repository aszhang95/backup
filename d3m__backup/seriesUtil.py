# seriesUtil

import pandas as pd
import numpy as np
from numpy import nan



def read_series(filename, delimiter, datatype=None):
    '''
    Helper method:
    Reads in file with mixed row lengths (timeseries of different length)
    (direction = horizontal)

    Inputs -
        filename (string): path to file
        delimiter (char): value delimiter
        datatype (Python data type or np.dtype)

    Outputs -
        tuple of pandas.DataFrame with missing values as NaN
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
    if datatype is not None:
        rv = pd.DataFrame(data, columns=range(max_col_len), dtype=datatype)
        rv.fillna(value=nan, inplace=True)
    else:
        rv = pd.DataFrame(data, columns=range(max_col_len),dtype=float)
        rv.fillna(value=nan, inplace=True)

    return rv




def write_series(input_data, filename=None):
    '''
    Helper function:
    Writes out input numpy.ndarray to file to interface with Data Smashing
        binary

    Inputs -
        input_data (np.ndarray): use quantized data (True) or original data
            (False)
        filename (string): name of output file

    Outputs -
        fh (NamedTemporaryFile): output filehandler
        out_fname (string): name of the output file
    '''

    if isinstance(input_data, np.ndarray):
        data = input_data.tolist()
    elif isinstance(input_data, pd.DataFrame):
        data = input_data.values.tolist()
    else:
        raise TypeError("Error: input_data is not np.nda or pd.DF")

    to_write = []
    for row in data:
        to_write.append( [int(x) for x in row if not pd.isnull(x)] )

    if filename is None:
        fh = tempfile.NamedTemporaryFile(dir=self.__file_dir, \
        delete=False)
        wr = csv.writer(fh, delimiter=" ")
        wr.writerows(to_write)
        fh.close()
        out_fname = path_leaf(fh.name)
        return fh, out_fname
    else:
        with open(self.__file_dir + "/" + filename, "w") as f:
            wr = csv.writer(f, delimiter=" ")
            wr.writerows(to_write)



def path_leaf(path):
    '''
    Helper function:
    Returns filename from a given path/to/file
        Taken entirely from Lauritz V. Thaulow on https://stackoverflow.com/questions/8384737

    Input -
        path (string): path/to/the/file

    Returns -
        filename (string)
    '''

    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)



def apply(f, input_data):
    '''
    Applies function to an input nda or df that may contain NaN's;
        maintains NaN's but applies functions to all other values
        and returns an output nda or df matching the input datatype

    Inputs -
        f (function): function to be applied to input data possibly containing
            NaN's
        input_data (np.nda or pd.DF): the input data

    Outputs -
        transformed np.nda or pd.DF
    '''

    input_type = None

    assert callable(f), "Error: input function not callable."
    if isinstance(input_data, np.ndarray):
        data = input_data.tolist()
        input_type = "nda"
    elif isinstance(input_data, pd.DataFrame):
        data = input_data.values.tolist()
        input_type = "df"
    else:
        raise TypeError("Input must be either pandas.DataFrame\
                        or numpy.ndarray.")
    total = []
    for row in data:
        total.append([ f(val) if not pd.isnull(val)\
                    else nan for val in row])
    if input_type == "nda":
        return np.array(total).astype(float)
    else:
        return pd.DataFrame(total, dtype=float)
