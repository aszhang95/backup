import numpy as np
import pandas as pd
from numpy import nan
import os
import csv



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



def write_series(input_data, out_path):
    '''
    Writes out np.nda or pd.DF to output file path

    Inputs -
        input_data (pd.DF or np.nda): data to be written out
        outpath (string): destination path

    Outputs -
        (None)
    '''

    if isinstance(input_data, pd.DataFrame):
        data = input_data.values.tolist()
    elif isinstance(input_data, np.ndarray):
        data = input_data.tolist()
    else:
        raise TypeError(\
        "Error: argument input_data not is not pandas.DataFrame or numpy.ndarray")

    to_write = []
    for row in data:
        to_write.append( [int(x) for x in row if not pd.isnull(x)] )
    with open(out_path, 'w') as out:
        wr = csv.writer(out, delimiter=" ")
        wr.writerows(to_write)

    print("Done!")
