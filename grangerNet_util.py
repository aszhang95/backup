from datetime import datetime
import csv
import pandas as pd
import numpy as np
from ast import literal_eval
import re


# def find_date_range(data_path, date_form="%m/%d/%Y"):
def find_date_range(data_path, date_col="Date", type_col="Primary Type", \
                    loc_col="Location", out_fname="data_formated.csv"):
    """
    Finds min/max date from input data_file

    Inputs -
        headers (boolean): whether the input data file has headers
            (makes preproc faster)
        date_form (string): if timedata uses different date formatting,
            needs to be specified
    Outputs -
        min_date, max_date (strings)
    """
    # would we have to deal w/ file w/o headers?

    # # reading and comparing potentially faster than sorting the df?
    # min_date = datetime.strptime("12/12/3000","%m/%d/%Y")
    # max_date = datetime.strptime("01/01/1900","%m/%d/%Y")
    # with open(data_path, "r") as input_data:
    #     for line in input_data:
    #         try:
    #             date = datetime.strptime(str(line), date_form)
    #         except ValueError:
    #             print("Error: couldn't find date")
    #             continue
    #
    #         print(date)
    #
    #         if date < min_date:
    #             min_date = date
    #         elif date > max_date:
    #             max_date = date

    data = pd.read_csv(data_path, usecols=[date_col, type_col, loc_col],\
                        parse_dates=["Date"])
    data[date_col] = pd.to_datetime(data[date_col], infer_datetime_format=True)
    data.sort_values(date_col, inplace=True)
    min_date = data.iloc[0][date_col]
    max_date = data.iloc[(data.shape[0]-1)][date_col]

    lat = []
    lon = []
    for row in data.itertuples(index=True, name='Pandas'):
        index = (row.Index)
        loc = literal_eval(getattr(row, loc_col))
        if "," in data.loc[index, type_col]:
            data.loc[index, type_col] = data.loc[index, type_col].replace(",", " ")
        try:
            lat.append(loc[0])
            lon.append(loc[1])
        except:
            lat.append(np.NaN)
            lon.append(np.NaN)
    data["Latitude"] = lat
    data["Longitude"] = lon
    data.drop(loc_col, axis=1, inplace=True)

    data.to_csv(out_fname, na_rep="", header=False, index=False)

    labelled = {'min_date': min_date, 'max_date': max_date, 'df':data}

    return labelled



def parse_loc(file_path):
    line_mapping = {}
    line_num = 0
    with open(file_path) as f:
        for line in f:
            loc_pair = re.findall(r'-?\d{1,2}\.\d{2,4}', line)
            # assumption is that first of pair is latitude and second of pair is
            # Longitude
            if len(loc_pair) > 0:
                lat = float(loc_pair[0])
                lon = float(loc_pair[1])
                if lat >= -90 and lat <= 90 and lon >= -180 and lon <=180:
                    line_mapping[line_num] = (lat, lon)
            line_num += 1
    rv = pd.DataFrame.from_dict(line_mapping, orient="index")
    return rv.rename(index=str, columns={0: "Latitude", 1: "Longitude"})



def parse_timeseries(file_path, min_date, max_date):
    data = []
    with open(file_path) as f:
        for line in f:
            # first value is the index, subsequent values are the timeseries
            ts = line.strip().split(" ")
            data.append(ts)
    # daterange = pd.date_range(min_date, max_date)
    daterange = pd.date_range(min_date, min_date)
    return pd.DataFrame(data, columns=(["index"]+list(daterange))).set_index("index")
