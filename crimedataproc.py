import datetime
import csv
import os
import uuid
import atexit
import subprocess as sp
import pandas as pd
from grangerNet import spatiotemporal_data_
from subprocess import Popen, PIPE
from ast import literal_eval
import cPickle as pickle


# global variables
CWD = os.getcwd()
TEMP_DIR = str(uuid.uuid4())
TEMP_DIR = TEMP_DIR.replace("-", "")



class CrimeData(spatiotemporal_data_):
    """
    Crime Prediction implementation of spatiotemporal_data_ class:

    class for reading in spatio temporal data, and provide it in a
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

    def __init__(self, path, bin_path, livepath=None):
        assert os.path.isfile(path), "Error: input file specified does not exist\
        or cannot be found!"
        self._path=path # path to the full dataset to be processed
        assert os.path.exists(bin_path), "Error: bin_path specified does not exist\
        or cannot be found!"
        self.__bin_path = bin_path
        self._livepath=livepath
        self._indexmap=None
        prev_wd = os.getcwd()
        os.chdir(CWD)
        sp.Popen("mkdir "+ TEMP_DIR, shell=True, stderr=sp.STDOUT).wait()
        self.__file_dir = CWD + "/" + TEMP_DIR
        os.chdir(prev_wd)
        self._meta_dict = None


    @property
    def path(self):
        return self._path


    @property
    def bin_path(self):
        return self.__bin_path


    @property
    def indexmap(self):
        return self._indexmap


    @property
    def file_dir(self):
        return self.__file_dir


    @property
    def meta_dict(self):
        return self._meta_dict


    @path.setter
    def path(self, new_path):
        assert os.path.isfile(new_path), "Error: File not found."
        self._path = new_path


    @bin_path.setter
    def bin_path(self, new_bin_path):
        self.__bin_path = os.path.abspath(new_bin_path)


    def pull(self, *arg, **kwds):
        """
        pull data from webpage

        Args:
            path

        Returns:
            log file and/or self object

        Raises:
            When errors or inaccessible
        """
        pass


    #def meta_properties(self, keywords_list, date_col="Date", type_col="Primary Type", \
    def meta_properties(self, date_col="Date", type_col="Primary Type", lat_col="Latitude",\
                        lon_col="Longitude", loc_col="Location", out_fname="data_formated.csv"):
        """compute meta properties from data
        which include: size and complexity of data set,
        location properties, time span, number of attributes if any

        Args:
            keyword specifying what property to compute
            headers (boolean): whether the input data file has headers
                (makes preproc faster)
            date_form (string): if timedata uses different date formatting,
                needs to be specified

        Returns:
            A dict mapping keywords to properties

        Raises:
            Unimplemeneted keyword. Might be a warning
        """
        # implement keywords
        # would we have to deal w/ file w/o headers?
        data = pd.read_csv(self._path, usecols=[date_col, type_col, lat_col, lon_col, loc_col],\
                            parse_dates=[date_col], infer_datetime_format=True)
        data.sort_values(date_col, inplace=True)
        min_date = data.iloc[0][date_col]
        max_date = data.iloc[(data.shape[0]-1)][date_col]

        lat = []
        lon = []

        nulls = []
        for row in data.itertuples(index=True, name='Pandas'):
            index = (row.Index)
            # if lat, lon = nan, drop the row
            # update: confirmed that issue is with code, not with data; for some reason
            # csv is actually correctly grabbing location, there just legitimately are
            # entries w/o location data
            if pd.isnull(getattr(row, loc_col)):
                # print("row: {} got a {} for the 'Location' column with date: {}".format(index, \
                # getattr(row, loc_col), getattr(row, date_col)))
                if not pd.isnull(getattr(row, lat_col)) and not pd.isnull(getattr(row, lon_col)):
                    lat.append(str(getattr(row, lat_col)))
                    lon.append(str(getattr(row, lon_col)))
                    if "," in data.loc[index, type_col]:
                        data.loc[index, type_col] = data.loc[index, type_col].replace(",", " ")
                    print(\
                    "Successfully extracted lat, lon from lat_col, lon_col for row: {}".format(index))
                else:
                    nulls.append((index, getattr(row, date_col)))
                    data.drop(index, inplace=True)
                    # print("No location data available for row: {} with date: {}".format(index,\
                    # getattr(row, date_col)))
            else:
                loc = literal_eval(getattr(row, loc_col))
                lat.append(loc[0])
                lon.append(loc[1])
                if "," in data.loc[index, type_col]:
                    data.loc[index, type_col] = data.loc[index, type_col].replace(",", " ")

        data["Latitude"] = lat
        data["Longitude"] = lon
        data.drop(loc_col, axis=1, inplace=True)

        data.sort_values("Latitude", inplace=True)
        min_lat = float(data.iloc[0]["Latitude"])
        max_lat = float(data.iloc[(data.shape[0]-1)]["Latitude"])

        data.sort_values("Longitude", inplace=True)
        min_lon = float(data.iloc[0]["Longitude"])
        max_lon = float(data.iloc[(data.shape[0]-1)]["Longitude"])

        data.to_csv(self.__file_dir+'/'+out_fname, na_rep="", header=False, index=False)

        attrs = {'min_date': min_date, 'max_date': max_date, "min_lat":min_lat,\
                "max_lat":max_lat, "min_lon":min_lon, "max_lon":max_lon, \
                "dates":pd.date_range(min_date, max_date), "num_attributes": data.shape[1],\
                "num_entries":data.shape[0]}
        self._meta_dict = attrs
        self._meta_dict['df'] = data
        pickle.dump(data, open(CWD + "/meta_dict.p", "wb"))
        print("Num entries w/o location data: {}".format(len(nulls)))
        pickle.dump(nulls, open(CWD + "/nulls.p", "wb"))

        # not include the formatted dataset?
        return attrs


    def parse_loc(self):
        line_mapping = {}
        line_num = 0
        with open("DATASTAT.dat") as f:
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


    def parse_timeseries(self, file_path):
        data = []
        with open(file_path) as f:
            for line in f:
                # first value is the index, subsequent values are the timeseries
                ts = line.strip().split(" ")
                data.append(ts)
        # daterange = pd.date_range(min_date, max_date)
        daterange = pd.date_range(self._meta_dict["min_date"], self._meta_dict["max_date"])
        return pd.DataFrame(data, columns=(["index"]+list(daterange))).set_index("index")


    def transform(self, grid_size=200, force=False, date_col="Date",
                type_col="Primary Type", loc_col="Location",
                out_fname="data_formated.csv"):
        """
        transforms tabular data to quantized time series corpora,
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

        if self._meta_dict is None or force:
            self.meta_properties(date_col, type_col, loc_col, out_fname)

        cwd = os.getcwd()
        os.cd(self.__file_dir)
        assert os.path.isfile(out_fname), "Error: Please try again using force=True."

        command = self.__bin_path + " " + out_fname + "%Y-%m-%d %H:%M:%S" + " "
        command += (self._meta_dict["min_lat"] + " " + self._meta_dict["max_lat"] + " ")
        command += (self._meta_dict["min_lon"] + " " + self._meta_dict["max_lat"] + " ")
        command += (grid_size + " ")
        if os.path.isfile("out_ts" + str(grid_size) + ".csv"):
                os.remove("out_ts" + str(grid_size) + ".csv")
        command += ("out_ts" + str(grid_size) + ".csv")
        sp.Popen(command, shell=True, stderr=sp.STDOUT).wait()

        assert os.path.isfile("out_ts" + str(grid_size) + ".csv") and \
        os.path.isfile("DATASTAT.dat"), "Error: please retry."

        loc_df = self.parse_loc()
        ts_df = self.parse_timeseries("out_ts" + str(grid_size) + ".csv")
        combined = pd.concat([loc_df, ts_df], axis=1)
        return combined


def cleanup():
    '''
    Maintenance function:
    Clean up library files before closing the script; no I/O
    '''

    prev_wd = os.getcwd()
    os.chdir(CWD)
    if os.path.exists(CWD + "/" + TEMP_DIR):
        command = "rm -r " + CWD + "/" + TEMP_DIR
        sp.Popen(command, shell=True, stderr=sp.STDOUT).wait()
    os.chdir(prev_wd)



atexit.register(cleanup)
