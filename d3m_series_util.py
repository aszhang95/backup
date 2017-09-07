# d3m_series_util

import pandas as pd
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
        rv = pd.DataFrame(data, columns=range(max_col_len))
        rv.fillna(value=nan, inplace=True)

    return rv
