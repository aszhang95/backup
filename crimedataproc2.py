import datetime
from dateutil.parser import parse
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
import re
import pdb
from copy import deepcopy
import numpy as np


# global variables
CWD = os.getcwd()
TEMP_DIR = str(uuid.uuid4())
TEMP_DIR = TEMP_DIR.replace("-", "")
date_pattern = re.compile('\d{4}-\d{2}-\d{2}')



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
        __bin_path: path to the bin/procdata
        _path:     path to data log
        _livepath: webpath to data interface
        _meta_dict: dict of meta properties
        _num_crimesites: number of unique Latitude/Longitude pairs
        _loc_ts_df: working pd.DF of lat/lon pair w/ timeseries mapped to dates
    """

    def __init__(self, path, bin_path, livepath=None, file_dir=None, \
                meta_dict_path=None):
        assert os.path.isfile(path), \
        "Error: input file specified does not exist or cannot be found!"
        self._path=os.path.abspath(path) # path to the full dataset to be processed
        assert os.path.exists(bin_path), \
        "Error: bin_path specified does not exist or cannot be found!"
        self._bin_path=os.path.abspath(bin_path)
        self._livepath = livepath
        if file_dir is None:
            prev_wd=os.getcwd()
            os.chdir(CWD)
            sp.Popen("mkdir "+ TEMP_DIR, shell=True, stderr=sp.STDOUT).wait()
            self._file_dir = CWD + "/" + TEMP_DIR
            os.chdir(prev_wd)
        else:
            self._file_dir = os.path.abspath(file_dir)
        if meta_dict_path is not None:
            assert os.path.isfile(meta_dict_path), \
            "Error: input file specified does not exist or cannot be found!"
            self._meta_dict = pickle.load(open(meta_dict_path, "rb" ))
            self._num_crimesites = self._meta_dict["num_crimesites"]
            self._loc_ts_df = self._meta_dict["loc_ts_df"]
            self._indexmap = self._meta_dict["indexmap"]
        else:
            self._meta_dict = None
            self._num_crimesites = 0
            self._loc_ts_df = None
            self._indexmap = None


    ATTRS = ["_indexmap", "_bin_path", "_path", "_livepath", \
            "_meta_dict", "num_crimesites"]


    @property
    def path(self):
        return self._path


    @property
    def bin_path(self):
        return self._bin_path

    @property
    def livepath(self):
        return self._livepath


    @property
    def indexmap(self):
        return self._indexmap


    @property
    def loc_ts_df(self):
        return self._loc_ts_df


    @property
    def file_dir(self):
        return self._file_dir


    @property
    def meta_dict(self):
        return self._meta_dict


    @property
    def num_crimesites(self):
        return self._num_crimesites


    @path.setter
    def path(self, new_path):
        assert os.path.isfile(new_path), "Error: File not found."
        self._path = os.path.abspath(new_path)


    @bin_path.setter
    def bin_path(self, new_bin_path):
        self._bin_path = os.path.abspath(new_bin_path)

    @livepath.setter
    def livepath(self, new_livepath):
        self._livepath = new_livepath


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


    def extract_date(self, string):
        matches = re.match(date_pattern, string)
        if matches:
            return parse(matches.group(0))
        else:
            return None


    #def meta_properties(self, keywords_list, date_col="Date", type_col="Primary Type", \
    def meta_properties(self, date_col="Date", type_col="Primary Type", lat_col="Latitude",\
                        lon_col="Longitude", out_fname="data_formated.csv", \
                        meta_dict_pickle_fname="meta_dict.p"):
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
        data = pd.read_csv(self._path, usecols=[date_col, type_col, lat_col, lon_col],\
                            parse_dates=[date_col], infer_datetime_format=True)
        data.dropna(axis=0, how="any", inplace=True)

        data.sort_values(date_col, inplace=True)
        min_date = data.iloc[0][date_col].date()
        max_date = data.iloc[-1][date_col].date()

        data.sort_values("Latitude", inplace=True)
        min_lat = float(data.iloc[0]["Latitude"])
        max_lat = float(data.iloc[(data.shape[0]-1)]["Latitude"])

        data.sort_values("Longitude", inplace=True)
        min_lon = float(data.iloc[0]["Longitude"])
        max_lon = float(data.iloc[(data.shape[0]-1)]["Longitude"])

        data.to_csv(self._file_dir+'/'+out_fname, na_rep="", header=False, index=False)

        self._meta_dict = {'min_date': min_date, 'max_date': max_date, "min_lat":min_lat,\
                "max_lat":max_lat, "min_lon":min_lon, "max_lon":max_lon,\
                "num_attributes": data.shape[1], "to_preproc_df": data}

        pickle.dump(self._meta_dict, open(self._file_dir + "/" + meta_dict_pickle_fname, "wb"))

        return self._meta_dict


    def parse_loc(self):
        '''
        Finds index, latitude, longitude pair from bin/proc data output DATA.STAT
        '''

        line_mapping = {}
        line_num = 0
        with open("DATASTAT.dat") as f:
            for line in f:
                loc_pair = re.findall(r'-?\d{1,2}\.\d{1,4}', line)
                # assumption is that first of pair is latitude and second of pair is
                # Longitude
                if len(loc_pair) > 0:
                    lat_start = float(loc_pair[0])
                    lat_stop = float(loc_pair[1])
                    lon_start = float(loc_pair[2])
                    lon_stop = float(loc_pair[3])
                    if -90 <= lat_start <= 90  and -90 <= lat_stop <= 90\
                    and -180 <= lon_start <= 180 and -180 <= lon_stop <= 180:
                        line_mapping[line_num] = (lat_start, lat_stop, lon_start, lon_stop)
                line_num += 1
        self._indexmap = pd.DataFrame.from_dict(line_mapping, orient="index")
        self._num_crimesites = line_num
        self._meta_dict["num_crimesites"] = self._num_crimesites

        self._indexmap.apply(pd.to_numeric, errors="ignore")
        self._indexmap.rename(index=int, columns={0: "Latitude_start", 1: "Latitude_stop",\
        2: "Longitude_start", 3: "Longitude_stop"}, inplace=True)

        pickle.dump(self._indexmap, open(self._file_dir + "/indexmap.p", "wb"))
        self._meta_dict["indexmap"] = self._indexmap

        return self._indexmap


    def parse_timeseries(self, file_path):
        '''
        Takes timeseries output from bin/procdata to create pd.DF of timeseries events
        for each index (location within grid)
        '''

        data = []
        with open(file_path) as f:
            for line in f:
                # first value is the index, subsequent values are the timeseries
                ts = line.strip().split(" ")
                data.append(ts)
        # daterange = pd.date_range(min_date, max_date)
        daterange = pd.date_range(self._meta_dict["min_date"], self._meta_dict["max_date"])
        self._meta_dict["daterange"] = [x.date() for x in daterange.tolist()]
        return pd.DataFrame(data, columns=(["index"]+self._meta_dict["daterange"]), dtype=int).set_index("index")


    def transform(self, grid_size=200, force=False, date_col="Date", type_col="Primary Type",\
                lat_col="Latitude", lon_col="Longitude",\
                out_fname="data_formated.csv", loc_ts_pickle_fname="timeseries_grid_data.p"):
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
            self.meta_properties(date_col, type_col, lat_col, lon_col, out_fname)

        self._meta_dict["delta_lat"]=(self._meta_dict["max_lat"]-self._meta_dict["min_lat"])/grid_size
        self.meta_dict["delta_lon"]=(self.meta_dict["max_lon"] - self._meta_dict["min_lon"])/grid_size
        self._meta_dict["grid_dims"] = grid_size

        cwd = os.getcwd()
        os.chdir(self._file_dir)
        assert os.path.isfile(out_fname), "Error: Please try again using force=True."

        command = self._bin_path + " " + out_fname + " '%Y-%m-%d %H:%M:%S'" + " "
        command += (str(self._meta_dict["min_lat"]) + " " + str(self._meta_dict["max_lat"]) + " ")
        command += (str(self._meta_dict["min_lon"]) + " " + str(self._meta_dict["max_lon"]) + " ")
        command += (str(grid_size) + " ")
        if os.path.isfile(str(grid_size)):
                os.remove(str(grid_size))
        command += (str(grid_size))

        print("Calling bin/dataproc to process input data.")
        print("Command: {}".format(command))
        sp.Popen(command, shell=True, stderr=sp.STDOUT).wait()

        print("bin/dataproc call complete; trying to read output.")
        assert os.path.isfile(str(grid_size)) and \
        os.path.isfile("DATASTAT.dat"), "Error: please retry."

        loc_df = self.parse_loc()
        ts_df = self.parse_timeseries(str(grid_size))
        self._meta_dict["loc_ts_df"] = \
        pd.concat([loc_df, ts_df], axis=1).sort_values(["Latitude_start", "Latitude_stop",\
        "Longitude_start", "Longitude_stop"]) # default is ascending = True

        os.chdir(cwd)

        pickle.dump(self._meta_dict["loc_ts_df"], \
        open(self._file_dir + "/" + loc_ts_pickle_fname, "wb"))

        self._loc_ts_df = self._meta_dict["loc_ts_df"]
        return self._loc_ts_df


    def export(self, path=None):
        '''
        Write out all relevant class attributes into dictionary to be exported using Pickle
        '''

        if path is None:
            path = self._file_dir + "/" + "meta_dict_export.p"
        pickle.dump(self._meta_dict, open(path, "wb"))


    def update(self, fpath, date_col="Date", lat_col="Latitude", lon_col="Longitude"):
        '''
        Reads in new lines from csv and update the database; assumes that the file has headers
        assumes an indexmap has been created

        checklist:
            - update daterange in meta_dict
            - update min/max date in meta_dict
            - update min/max lat/lon in meta_dict
            - update indexmap w/ new indices
            - update num_crimesites w/n meta_dict

        Inputs -
            fpath (string): path to new file to be processed

        Outputs -
            (modifies class in place)
        '''
        # NOTE: FOR INCREASED EFFICIENCY, CAN CHECK TO SEE WHETHER DATA EXISTS
        # IN THE DATASET ALREADY???

        # only need to be accessed once; don't change or are incremented by the method
        delta_lat, delta_lon = self._meta_dict["delta_lat"], self._meta_dict["delta_lon"]
        last_index = self._loc_ts_df.index[-1]

        data = pd.read_csv(fpath, usecols=[date_col, lat_col, lon_col],\
                            parse_dates=[date_col], infer_datetime_format=True)
        data.dropna(axis=0, how="any", inplace=True)
        print("Input data: ", data)

        # pdb.set_trace()
        for row in data.itertuples(index=True, name='Pandas'):
            min_lat, max_lat, min_lon, max_lon = self._meta_dict["min_lat"], self._meta_dict["max_lat"],\
            self._meta_dict["min_lon"], self._meta_dict["max_lon"]
            min_date, max_date = self._meta_dict["min_date"], self._meta_dict["max_date"]
            row_date = getattr(row, date_col).date()
            row_lat = getattr(row, lat_col)
            row_lon = getattr(row, lon_col)
            # case 1: either lat or lon or both are outside of the max range for location
            if row_lat < min_lat or row_lat > max_lat or row_lon < min_lon or row_lon > max_lon:
                num_reps = self.incr_date(row_date)
                if num_reps != 0: # df was extended so value is actually significant
                    new_row = [(row_lat-delta_lat/2), (row_lat+delta_lat/2), (row_lon-delta_lon/2),\
                    (row_lon+delta_lon/2)] + [0]*num_reps
                else: # df wasn't extended so value isn't significant
                    new_row = [(row_lat-delta_lat/2), (row_lat+delta_lat/2), (row_lon-delta_lon/2),\
                    (row_lon+delta_lon/2)] + [0]*(self._loc_ts_df.shape[1]-4)
                last_index += 1
                self.loc_ts_df.loc[last_index] = new_row
                # if cared about how many crimes happened at this date, then would increment
                self._loc_ts_df.loc[last_index, row_date] = 1
                # updating meta_properties of proc
                self._indexmap.loc[last_index] = [(row_lat-delta_lat/2), (row_lat+delta_lat/2), (row_lon-delta_lon/2),\
                (row_lon+delta_lon/2)]
                self._meta_dict["indexmap"].loc[last_index] = self._indexmap.loc[last_index]
                self._meta_dict["loc_ts_df"] = self._loc_ts_df
                self._num_crimesites += 1
                self._meta_dict["num_crimesites"] = self._num_crimesites
                if row_lat < self._meta_dict["min_lat"]:
                    self._meta_dict["min_lat"] = row_lat
                elif row_lat > self._meta_dict["max_lat"]:
                    self._meta_dict["max_lat"] = row_lat
                if row_lon < self._meta_dict["min_lon"]:
                    self._meta_dict["min_lon"] = row_lon
                elif row_lon > self._meta_dict["max_lon"]:
                    self.meta_dict["max_lon"] = row_lon
            else: # case 2: lat/lon pair is w/n existing range
                # case 2.1 and is conjoint
                updated = False
                # always need to increment date
                num_reps = self.incr_date(row_date)
                for search_row in self._loc_ts_df.itertuples():
                    lat_lb, lat_ub = search_row[1], search_row[2]
                    lon_lb, lon_ub = search_row[3], search_row[4]
                    if lat_lb <= row_lat <= lat_ub and lon_lb <= row_lon <= lon_ub:
                        self._loc_ts_df.loc[search_row[0], row_date] = 1
                        self._meta_dict["loc_ts_df"] = self._loc_ts_df
                        self._num_crimesites += 1
                        self._meta_dict["num_crimesites"] = self._num_crimesites
                        updated = True
                        break
            # case 2.2 not conjoint
            # NOTE: non-conjoint could possibly overlap with existing grids
                if not updated:
                    num_reps = self.incr_date(row_date)
                    if num_reps != 0: # df was extended so value is actually significant
                        new_row = [(row_lat-delta_lat/2), (row_lat+delta_lat/2), (row_lon-delta_lon/2),\
                        (row_lon+delta_lon/2)] + [0]*num_reps
                    else: # df wasn't extended so value isn't significant
                        new_row = [(row_lat-delta_lat/2), (row_lat+delta_lat/2), (row_lon-delta_lon/2),\
                        (row_lon+delta_lon/2)] + [0]*(self._loc_ts_df.shape[1]-4)
                    last_index += 1
                    self.loc_ts_df.loc[last_index] = new_row
                    self._loc_ts_df.loc[last_index, row_date] = 1
                    # updating meta_properties
                    self._indexmap.loc[last_index] = [(row_lat-delta_lat/2), (row_lat+delta_lat/2), (row_lon-delta_lon/2),\
                    (row_lon+delta_lon/2)]
                    self._meta_dict["indexmap"].loc[last_index] = self._indexmap.loc[last_index]
                    self._meta_dict["loc_ts_df"] = self._loc_ts_df
                    self._num_crimesites += 1
                    self._meta_dict["num_crimesites"] = self._num_crimesites

        return self._loc_ts_df


    def incr_date(self, date):
        '''
        Helper function that extends the daterange of self._loc_ts_df columns if necessary;
        modifies input self._loc_ts_df in place

        Input -
            date (DateTime)

        Outputs -
            (int) number of dates w/n updated DataFrame
        '''

        if isinstance(date, str):
            date = datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S').date()

        extended = False
        if date < self._meta_dict["min_date"]: # self._loc_ts_df.columns[4] needs to be this new date
            daterange = pd.date_range(date, self._meta_dict["min_date"])
            self._meta_dict["daterange"] = [x.date() for x in daterange.tolist()]
            front = self._loc_ts_df.columns.tolist()[0:4]
            back = self.loc_ts_df.columns.tolist()[5:]
            self._loc_ts_df = self._loc_ts_df.reindex(\
            columns=(front+self._meta_dict["daterange"]+back), fill_value=0)
            # prefix = np.zeros(shape=(self._loc_ts_df.shape[0], len(daterange)))
            # pre_df = pd.DataFrame(prefix, columns=list(daterange), \
            # index=self._loc_ts_df.index.tolist())
            self._meta_dict["min_date"] = date
            self.meta_dict["loc_ts_df"] = self._loc_ts_df
            extended = True
        elif date > self._meta_dict["max_date"]:
            daterange = pd.date_range(self._meta_dict["max_date"], date)
            self._meta_dict["daterange"] = [x.date() for x in daterange.tolist()]
            self._loc_ts_df = self._loc_ts_df.reindex(\
            columns=(self._loc_ts_df.columns.tolist()+self._meta_dict["daterange"][1:]), fill_value=0)
            self._meta_dict["max_date"] = date
            self.meta_dict["loc_ts_df"] = self._loc_ts_df
            extended = True
        # else: date w/n daterange
        if extended:
            return self.loc_ts_df.shape[1]-4 # first two columns are lat/lon
        else:
            return 0


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



# atexit.register(cleanup)
