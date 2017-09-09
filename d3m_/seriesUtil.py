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
