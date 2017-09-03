# will need mapping dictionary, but the keys are boundaries
# need to read in every file from a folder

import os
import pandas as pd
from numpy import nan
import numpy as np

def read_in_vert(directory, label=None, bound=None, lb=None, ub=None, dtype_=float):
    '''
    Converts folder of libraries with vertical orientation to pd.DataFrame for
        use with SMB and smashmatch - takes all the files within the folder to be
        row examples of a class

    Inputs -
        directory (string): path to file
        delimiter (char): value delimiter
        datatype (type): type of values read into pd.DataFrame
        bound (int or float)
        lb (int or float): value to reassign input values if they are
            less than bound (must have bound to have lb and ub)
        ub (int or float): value to reassign input values if they are
            greater than bound (must have bound to have lb and ub)

    Outputs -
        tuple of pandas.DataFrame with missing values as NaN and label or just
            pd.DataFrame if no bound given (to feed into SMB.condense())
    '''

    if bound is not None:
        assert (lb is not None and ub is not None), "Error: cannot specify bound\
        without specifying quantization"
    if os.path.isdir(directory):
        total = []
        max_cols = 0
        if bound is not None:
            for filename in os.listdir(directory):
                row = []
                with open(directory+"/"+filename, "r") as f:
                    col_counter = 0
                    for line in f:
                        val = float(line.rstrip())
                        if val < bound:
                            row.append(lb)
                        elif val > bound:
                            row.append(ub)
                        else:
                            row.append(val)
                        col_counter += 1
                if col_counter > max_cols:
                    max_cols = col_counter
                total.append(row)
        else:
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
        rv = pd.DataFrame(total, columns=range(max_cols), dtype=dtype_)
        rv.fillna(value=nan, inplace=True)
        if label is not None:
            return (rv, label)
        else:
            return rv
    else: # path to a singular file
        row = []
        col_counter = 0
        if bound is not None:
            with open(directory+"/"+filename, "r") as f:
                for line in f:
                    val = float(line.rstrip())
                    if val < bound:
                        row.append(lb)
                    elif val > bound:
                        row.append(ub)
                    else:
                        row.append(val)
                    col_counter += 1
        else:
            with open(directory+"/"+filename, "r") as f:
                for line in f:
                    row.append(line.strip())
                    col_counter += 1
        rv = pd.DataFrame(row, columns=range(col_counter), dtype=dtype_)
        rv.fillna(value=nan, inplace=True)
        return rv


def read_in_ragged(filename, delimiter_, datatype=int, bound=None, lb=None, up=None):
    '''
    Reads in file with mixed column lengths (timeseries of different length)

    Inputs -
        file (string): path to file
        delimiter (char): value delimiter
        datatype (type): type of values read into pd.DataFrame
        bound (int or float)
        lb (int or float): value to reassign input values if they are
            less than bound (must have bound to have lb and ub)
        ub (int or float): value to reassign input values if they are
            greater than bound (must have bound to have lb and ub)

    Outputs -
        tuple of pandas.DataFrame with missing values as NaN and label or just
            pd.DataFrame if no bound given (to feed into SMB.condense())
    '''

    if bound is not None:
        assert (lb is not None and up is not None), "Error: cannot specify boundary\
        without specifying quantization"
    data = []
    max_col_len = 0
    with open(filename, 'r') as f:
        if bound is not None:
            for line in f:
                formatted_row = []
                row = line.strip('\n').split(delimiter_)
                if row[-1] == "":
                    row = row[:-1]
                row_len = len(row)
                if row_len > max_col_len:
                    max_col_len = row_len
                for val in row:
                    fval = float(val)
                    if fval < bound:
                        formatted_row.append(lb)
                    elif fval > bound:
                        formatted_row.append(ub)
                    else:
                        formatted_row.append(fval)
                data.append(formatted_row)
        else:
            for line in f:
                formatted_row = []
                row = line.strip('\n').split(delimiter_)
                if row[-1] == "":
                    row = row[:-1]
                row_len = len(row)
                if row_len > max_col_len:
                    max_col_len = row_len
                data.append(row)
    rv = pd.DataFrame(data, columns=range(max_col_len), dtype=datatype)
    rv.fillna(value=nan, inplace=True)
    return rv


    def write_out_ragged(path, df_):
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
            to_write.append([int(x for x in row if not pd.isnull(x)])
        with open(path, 'w') as f:
            wr = csv.writer(f, delimiter=" ")
            wr.writerows(to_write)
