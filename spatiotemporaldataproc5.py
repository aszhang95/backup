import datetime
from dateutil.parser import parse
import csv
import os
import atexit
import tempfile
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
import dateutil.parser as dparser
import shutil
import warnings



class SpatialTemporalData(spatiotemporal_data_):
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
           accessed by method @dataset_properties@

        3. Spatio-temporal quantization method  @transform_with_binary@. Returns ndarray, indexmap

        4. Pull data from web interface specified in livepath by @pull@

        5. Legacy support for existing data directory.

    Attributes:
        _indexmap: map from data index to location
        __bin_path: path to the bin/procdata
        _path:     path to data log
        _livepath: webpath to data interface
        _data_properties_dict: dict of meta properties
        _num_entries: number of unique Latitude/Longitude pairs
        _dataset_df: working pd.DF of lat/lon pair w/ timeseries mapped to dates
    """

    def __init__(self, path, bin_path, livepath=None, file_dir=None, \
                data_properties_dict_path=None):
        assert os.path.isfile(path), \
        "Error: input file specified does not exist or cannot be found!"
        self._path=os.path.abspath(path) # path to the full dataset to be processed
        assert os.path.exists(bin_path), \
        "Error: bin_path specified does not exist or cannot be found!"
        self._bin_path=os.path.abspath(bin_path)
        self._livepath = livepath
        if file_dir is None:
            self._file_dir = tempfile.mkdtemp(prefix="spatiotemporaldataproctemp_")
        else:
            self._file_dir = os.path.abspath(file_dir)
        if data_properties_dict_path is not None:
            assert os.path.isfile(data_properties_dict_path), \
            "Error: input file specified does not exist or cannot be found!"
            self._data_properties_dict = pickle.load(open(data_properties_dict_path, "rb" ))
            self._num_entries = self._data_properties_dict["num_entries"]
            self._dataset_df = self._data_properties_dict["dataset_df"]
            self._indexmap = self._data_properties_dict["indexmap"]
        else:
            self._data_properties_dict = None
            self._num_entries = 0
            self._dataset_df = None
            self._indexmap = None


    ATTRS = ["_indexmap", "_bin_path", "_path", "_livepath", \
            "_data_properties_dict", "num_entries"]


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
    def dataset_df(self):
        return self._dataset_df


    @property
    def file_dir(self):
        return self._file_dir


    @property
    def data_properties_dict(self):
        return self._data_properties_dict


    @property
    def num_entries(self):
        return self._num_entries


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


    def meta_properties(self):
        '''
        Compute meta properties from data
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
        '''
        pass


    def dataset_properties(self, date_col="Date", type_col="Primary Type", lat_col="Latitude",\
                        lon_col="Longitude", out_fname="data_formated.csv", \
                        data_properties_dict_pickle_fname="data_properties_dict.p", type_list=None,\
                        id_col="ID"):
        """
        Preliminary method to gather information about dataset; originally implemented
        as meta_properties method but for clearer naming, method name was changed

        """
        # implement keywords
        # would we have to deal w/ file w/o headers?
        data = pd.read_csv(self._path, usecols=[date_col, type_col, lat_col, lon_col, id_col],\
                            parse_dates=[date_col], infer_datetime_format=True)
        data.dropna(axis=0, how="any", inplace=True)

        rejected_types = set()
        if type_list is not None:
            indices_to_drop = []
            for row in data.itertuples(index=True, name='Pandas'):
                entry_type = row[list(data.columns).index(type_col)+1] # universal that indices off by 1?
                if not str(entry_type) in type_list:
                    rejected_types.add(entry_type)
                    indices_to_drop.append(row[0]) # do not modify df while looping through
            data = data.drop(indices_to_drop)

        data.sort_values(date_col, inplace=True)
        print("Uncleaned large dataset \n: {}".format(data))
        print("Shape of uncleaned dataset: {}".format(data.shape))
        print("rejected_types: {}".format(rejected_types))
        min_date = data.iloc[0][date_col].date()
        max_date = data.iloc[-1][date_col].date()

        data.sort_values(lat_col, inplace=True)
        min_lat = float(data.iloc[0][lat_col])
        max_lat = float(data.iloc[(data.shape[0]-1)][lat_col])
        lat_precision = len(str(min_lat).split('.')[1])

        data.sort_values(lon_col, inplace=True)
        min_lon = float(data.iloc[0][lon_col])
        max_lon = float(data.iloc[(data.shape[0]-1)][lon_col])
        lon_precision = len(str(min_lon).split('.')[1])

        # have to drop ID column bc preproc doesn't expect that column to exist
        data_no_id_col = deepcopy(data).drop(labels=id_col, axis=1)
        data_no_id_col.to_csv(self._file_dir+'/'+out_fname, na_rep="", header=False, index=False)
        data = data.sort_values([lat_col, lon_col])

        self._data_properties_dict = {'min_date': min_date, 'max_date': max_date, "min_lat":min_lat,\
                "max_lat":max_lat, "min_lon":min_lon, "max_lon":max_lon,\
                "num_attributes": data.shape[1], "to_preproc_df": data, \
                "lat_precision": lat_precision, "lon_precision": lon_precision}
        self._data_properties_dict["accepted_types"] = set(type_list)
        self._data_properties_dict["rejected_types"] = rejected_types

        pickle.dump(self._data_properties_dict, open(self._file_dir + "/" + data_properties_dict_pickle_fname, "wb"))

        return self._data_properties_dict


    def parse_loc(self):
        '''
        Finds index, latitude, longitude pair from bin/proc data output DATA.STAT
        '''

        line_mapping = {}

        # line_num = 0
        # with open("DATASTAT.dat") as f:
        #     for line in f:
        #         loc_pair = re.findall(r'-?\d{1,2}\.\d{1,4}', line)
        #         if len(loc_pair) == 4:
        #             lat_start = float(loc_pair[0])
        #             lat_stop = float(loc_pair[1])
        #             lon_start = float(loc_pair[2])
        #             lon_stop = float(loc_pair[3])
        #             if -90 <= lat_start <= 90  and -90 <= lat_stop <= 90\
        #             and -180 <= lon_start <= 180 and -180 <= lon_stop <= 180:
        #                 line_mapping[line_num] = (lat_start, lat_stop, lon_start, lon_stop)
        #                 line_num += 1
        #         else:
        #             continue

        df = pd.read_csv("DATASTAT.dat", header=None, sep=" ")
        line_num = 0
        for row in df.itertuples(index=True, name='Pandas'):
            line_mapping[line_num] = [float(x) for x in row[1].split("#")]
            line_num += 1
        self._indexmap = pd.DataFrame.from_dict(line_mapping, orient="index")
        self._num_entries = line_num
        self._data_properties_dict["num_entries"] = self._num_entries

        self._indexmap.apply(pd.to_numeric, errors="ignore")
        self._indexmap.rename(index=int, columns={0: "Latitude_start", 1: "Latitude_stop",\
        2: "Longitude_start", 3: "Longitude_stop"}, inplace=True)

        pickle.dump(self._indexmap, open(self._file_dir + "/indexmap.p", "wb"))
        self._data_properties_dict["indexmap"] = self._indexmap

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
        daterange = pd.date_range(self._data_properties_dict["min_date"], self._data_properties_dict["max_date"])
        self._data_properties_dict["daterange"] = [x.date() for x in daterange.tolist()]
        return pd.DataFrame(data, columns=(["index"]+self._data_properties_dict["daterange"]), dtype=int).set_index("index")


    def transform_with_binary(self, grid_size=200, force=False, date_col="Date", type_col="Primary Type",\
                lat_col="Latitude", lon_col="Longitude", export_df_name="grid_df_export.csv",\
                out_fname="grid_ts_export.csv", type_list=None, id_col="ID"):
        """
        transforms tabular data to quantized time series corpora,
        given the spatial and temporal quantization parameters
        now also includes column for Event type

        Args:
            data: dataframe?
            spatial quantization parameters
            temporal quantization parameters

        Returns:
            quantized time series corpora, O
            indexmap

        Raises:
            When quantization specified is impossible
        """

        if self._data_properties_dict is None or force:
            self.dataset_properties(date_col, type_col, lat_col, lon_col, out_fname, type_list=type_list)

        self._data_properties_dict["delta_lat"]=(self._data_properties_dict["max_lat"]-self._data_properties_dict["min_lat"])/grid_size
        self.data_properties_dict["delta_lon"]=(self.data_properties_dict["max_lon"] - self._data_properties_dict["min_lon"])/grid_size
        self._data_properties_dict["grid_dims"] = grid_size

        cwd = os.path.abspath(os.getcwd())
        os.chdir(self._file_dir)
        assert os.path.isfile(out_fname), "Error: Please try again using force=True."

        command = self._bin_path + " " + out_fname + " '%Y-%m-%d %H:%M:%S'" + " "
        command += (str(self._data_properties_dict["min_lat"]) + " " + str(self._data_properties_dict["max_lat"]) + " ")
        command += (str(self._data_properties_dict["min_lon"]) + " " + str(self._data_properties_dict["max_lon"]) + " ")
        command += (str(grid_size) + " ")
        if os.path.isfile(str(grid_size)):
                os.remove(str(grid_size))
        command += (str(grid_size))
        # command += (" >& /dev/null")

        print("Calling bin/dataproc to process input data.")
        print("Command: {}".format(command))
        sp.Popen(command, shell=True, stderr=sp.STDOUT).wait()

        print("bin/dataproc call complete; trying to read output.")
        assert os.path.isfile(str(grid_size)) and \
        os.path.isfile("DATASTAT.dat"), "Error: please retry."

        loc_df = self.parse_loc()
        ts_df = self.parse_timeseries(str(grid_size))
        types_col = {"Event_type":[set() for _ in xrange(loc_df.shape[0])]}
        print("Created loc_df and ts_df; beginning to create types_df")
        type_df = pd.DataFrame.from_dict(types_col, dtype=object)
        # print("lat/lon df: {}".format(loc_df))
        # print("time series df: {}".format(ts_df))
        # print("type_df: {}".format(type_df))
        # return type_df, loc_df, ts_df
        self._dataset_df = pd.concat([type_df, loc_df, ts_df], axis=1)
        self._dataset_df = self._dataset_df.sort_values(\
        ["Latitude_start", "Latitude_stop", "Longitude_start", "Longitude_stop"])
        # print(self._dataset_df)
        # print("Beginning to update types column")
        for dataset_row in self._dataset_df.itertuples(index=True, name="Pandas"):
            lat_lb, lat_ub = dataset_row[2], dataset_row[3]
            lon_lb, lon_ub = dataset_row[4], dataset_row[5]

            match = self._data_properties_dict["to_preproc_df"].loc[\
            (self._data_properties_dict["to_preproc_df"][lat_col] >= lat_lb) &\
            (self._data_properties_dict["to_preproc_df"][lat_col] < lat_ub) &\
            (self._data_properties_dict["to_preproc_df"][lon_col] >= lon_lb) &\
            (self._data_properties_dict["to_preproc_df"][lon_col] < lon_ub)]

            self._dataset_df.at[dataset_row[0], "Event_type"] = set(match[type_col].unique())
            # print("data set row {} has types {}".format(dataset_row[0], self._dataset_df.iloc[[dataset_row[0]]]["Event_type"]))

        os.chdir(cwd)

        self._data_properties_dict["dataset_df"] = self._dataset_df
        # self._dataset_df.to_csv("../../../../../../project2/ishanu/CRIME/data/"+out_fname, na_rep="")
        self._dataset_df.to_csv(path_or_buf=export_df_name, na_rep="")

        return self._dataset_df


    def transform(self, grid_size=None, force=False, date_col="Date", type_col="Primary Type",\
                lat_col="Latitude", lon_col="Longitude", out_fname="data_formated.csv", \
                loc_ts_pickle_fname="timeseries_grid_data.p", type_list=None, lat_start=None,
                lat_stop=None, lon_start=None, lon_stop=None, freq='D'):
        '''
        Transform without C++ binary; not finished

        Bug:
            - indexmap not of the correct dimensions? (Doesn't match num rows of
              binary-produced ts)
            - gets caught in infinite loop? takes too long?
        '''

        if self._data_properties_dict is None or force:
            self.dataset_properties(date_col, type_col, lat_col, lon_col, out_fname, type_list=type_list)

        lat_start = []
        lat_stop = []
        lon_start = []
        lon_stop = []

        if lat_start is not None and lon_start is not None and lon_start is not None and lon_stop is not None:
            self._data_properties_dict["delta_lat"]=(lat_stop-lat_start)/grid_size
            self.data_properties_dict["delta_lon"]=(lon_stop-lon_start)/grid_size
            self._data_properties_dict["min_lat"] = lat_start
            self._data_properties_dict["max_lat"] = lat_stop
            self._data_properties_dict["min_lon"] = lon_start
            self._data_properties_dict["max_lon"] = lon_stop
        else:
            self._data_properties_dict["delta_lat"]=(self._data_properties_dict["max_lat"]-self._data_properties_dict["min_lat"])/grid_size
            self.data_properties_dict["delta_lon"]=(self.data_properties_dict["max_lon"]-self._data_properties_dict["min_lon"])/grid_size
        self._data_properties_dict["grid_dims"] = grid_size

        curr_lat = self._data_properties_dict["min_lat"]
        curr_lon = self._data_properties_dict["min_lon"]
        # creating the lat, lon pairs
        for i in range(grid_size):
            for j in range(grid_size):
                lat_start.append(curr_lat)
                lat_stop.append(curr_lat + self._data_properties_dict["delta_lat"])
                lon_start.append(curr_lon)
                lon_stop.append(curr_lon + self._data_properties_dict["delta_lon"])
                curr_lon += self._data_properties_dict["delta_lon"]
            curr_lat += self._data_properties_dict["delta_lat"]
            curr_lon = self._data_properties_dict["min_lon"]

        type_col_ = [set() for _ in xrange(len(lat_start))]

        grid_df = pd.DataFrame.from_dict({"Event_type": type_col_, "Latitude_start":lat_start, "Latitude_stop":lat_stop, \
        "Longitude_start":lon_start, "Longitude_stop":lon_stop})
        grid_df.index.name = "gridsquare_id"

        self._indexmap = grid_df.loc[:, grid_df.columns != 'Event_type']

        # create date columns of dataset DF
        daterange = pd.date_range(self._data_properties_dict["min_date"], self._data_properties_dict["max_date"], freq=freq)
        self._data_properties_dict["daterange"] = [x.date() for x in daterange.tolist()]
        zeros = np.zeros(shape=(len(lat_start), len(self._data_properties_dict["daterange"])))
        ts_df = pd.DataFrame(zeros, columns=self._data_properties_dict["daterange"], dtype=int)

        self._dataset_df = pd.concat([df_header, ts_df], axis=1)

        # now have to loop through raw data_frame to create dataset_df
        for dataset_row in self._dataset_df.itertuples(index=True, name="Pandas"):
            lat_lb, lat_ub = dataset_row[2], dataset_row[3]
            lon_lb, lon_ub = dataset_row[4], dataset_row[5]

            # Note: upperbound not inclusive
            match = self._data_properties_dict["to_preproc_df"].loc[\
            (self._data_properties_dict["to_preproc_df"][lat_col] >= lat_lb) &\
            (self._data_properties_dict["to_preproc_df"][lat_col] < lat_ub) &\
            (self._data_properties_dict["to_preproc_df"][lon_col] >= lon_lb) &\
            (self._data_properties_dict["to_preproc_df"][lon_col] < lon_ub)]

            self._dataset_df.at[dataset_row[0], "Event_type"] = set(match[type_col].unique())
            # now have to loop through match dataset to update timeseries
            for match_row in match.itertuples(index=True, name="Pandas"):
                self._dataset_df.at[dataset_row[0], match[date_col].date()] = 1

        df = deepcopy(self._dataset_df)
        df["sum_1"] = df.apply(lambda row: sum(row[:] == 0), axis = 1)
        Print("All entries of preproc_df are in df['sum_!'].sum() == self._data_properties_dict["to_preproc_df"].shape[0]
        return self._dataset_df


    def export(self, path=None):
        '''
        Write out all relevant class attributes into dictionary to be exported using Pickle
        '''

        if path is None:
            path = self._file_dir + "/" + "data_properties_dict_export.p"
        pickle.dump(self._data_properties_dict, open(path, "wb"))


    def update(self, fpath, date_col="Date", lat_col="Latitude", lon_col="Longitude",\
               type_col="Primary Type", type_list=None, id_col="ID"):
        '''
        Reads in new lines from csv and update the database; assumes that the file has headers
        assumes an indexmap has been created

        checklist:
            - update daterange in data_properties_dict
            - update min/max date in data_properties_dict
            - update min/max lat/lon in data_properties_dict
            - update indexmap w/ new indices
            - update num_entries w/n data_properties_dict

        Inputs -
            fpath (string): path to new file to be processed
            type_list (list of strings): list of desired entry types

        Outputs -
            (modifies class in place)
        '''

        # NOTE: FOR INCREASED EFFICIENCY, CAN CHECK TO SEE WHETHER DATA EXISTS
        # IN THE DATASET ALREADY???

        # only need to be accessed once; don't change or are incremented by the method
        delta_lat, delta_lon = self._data_properties_dict["delta_lat"], self._data_properties_dict["delta_lon"]
        last_index = self._dataset_df.index[-1]

        data = pd.read_csv(fpath, usecols=[date_col, type_col, lat_col, lon_col, id_col],\
                            parse_dates=[date_col], infer_datetime_format=True)
        data.dropna(axis=0, how="any", inplace=True)
        print("Input data: ", data)

        rejected_types = set()
        # pdb.set_trace()
        for row in data.itertuples(index=True, name='Pandas'):
            min_lat, max_lat, min_lon, max_lon = self._data_properties_dict["min_lat"], self._data_properties_dict["max_lat"],\
            self._data_properties_dict["min_lon"], self._data_properties_dict["max_lon"]
            min_date, max_date = self._data_properties_dict["min_date"], self._data_properties_dict["max_date"]
            row_date = getattr(row, date_col).date()
            row_lat = getattr(row, lat_col)
            row_lon = getattr(row, lon_col)
            row_type = str(getattr(row, type_col))
            # filter row first
            entry_type = getattr(row, type_col)
            if type_list is not None and not str(entry_type) in type_list:
                rejected_types.add(entry_type)
                continue # skip processing this entry b/c event type is of rejected event type

            # case 1: either lat or lon or both are outside of the max range for location
            if row_lat < min_lat or row_lat > max_lat or row_lon < min_lon or row_lon > max_lon:
                types = set()
                num_reps = self.incr_date(row_date)
                if num_reps != 0: # df was extended so value is actually significant
                    new_row = [types, (row_lat-delta_lat/2), (row_lat+delta_lat/2), (row_lon-delta_lon/2),\
                    (row_lon+delta_lon/2)] + [0]*num_reps
                else: # df wasn't extended so value isn't significant
                    new_row = [types, (row_lat-delta_lat/2), (row_lat+delta_lat/2), (row_lon-delta_lon/2),\
                    (row_lon+delta_lon/2)] + [0]*(self._dataset_df.shape[1]-5)
                last_index += 1
                self.dataset_df.loc[last_index] = new_row
                # if cared about how many crimes happened at this date, then would increment
                self._dataset_df.loc[last_index, row_date] = 1
                self._dataset_df.loc[last_index, "Event_type"].add(row_type)
                # updating dataset_properties of proc
                self._indexmap.loc[last_index] = [(row_lat-delta_lat/2), (row_lat+delta_lat/2), (row_lon-delta_lon/2),\
                (row_lon+delta_lon/2)]
                self._data_properties_dict["indexmap"].loc[last_index] = self._indexmap.loc[last_index]
                self._data_properties_dict["dataset_df"] = self._dataset_df
                self._num_entries += 1
                self._data_properties_dict["num_entries"] = self._num_entries
                if row_lat < self._data_properties_dict["min_lat"]:
                    self._data_properties_dict["min_lat"] = row_lat
                elif row_lat > self._data_properties_dict["max_lat"]:
                    self._data_properties_dict["max_lat"] = row_lat
                if row_lon < self._data_properties_dict["min_lon"]:
                    self._data_properties_dict["min_lon"] = row_lon
                elif row_lon > self._data_properties_dict["max_lon"]:
                    self.data_properties_dict["max_lon"] = row_lon
            else: # case 2: lat/lon pair is w/n existing range
                # case 2.1 and is conjoint
                updated = False
                # always need to increment date
                num_reps = self.incr_date(row_date)
                for search_row in self._dataset_df.itertuples():
                    lat_lb, lat_ub = search_row[1], search_row[2]
                    lon_lb, lon_ub = search_row[3], search_row[4]
                    if lat_lb <= row_lat <= lat_ub and lon_lb <= row_lon <= lon_ub:
                        self._dataset_df.loc[search_row[0], row_date] = 1
                        self._dataset_df.loc[search_row[0], "Event_type"].add(row_type)
                        self._data_properties_dict["dataset_df"] = self._dataset_df
                        self._num_entries += 1
                        self._data_properties_dict["num_entries"] = self._num_entries
                        updated = True
                        break
            # case 2.2 not conjoint
            # NOTE: non-conjoint could possibly overlap with existing grids
                if not updated:
                    types = set()
                    num_reps = self.incr_date(row_date)
                    if num_reps != 0: # df was extended so value is actually significant
                        new_row = [types, (row_lat-delta_lat/2), (row_lat+delta_lat/2), (row_lon-delta_lon/2),\
                        (row_lon+delta_lon/2)] + [0]*num_reps
                    else: # df wasn't extended so value isn't significant
                        new_row = [types, (row_lat-delta_lat/2), (row_lat+delta_lat/2), (row_lon-delta_lon/2),\
                        (row_lon+delta_lon/2)] + [0]*(self._dataset_df.shape[1]-5)
                    last_index += 1
                    self.dataset_df.loc[last_index] = new_row
                    self._dataset_df.loc[last_index, row_date] = 1
                    # updating dataset_properties
                    self._indexmap.loc[last_index] = [(row_lat-delta_lat/2), (row_lat+delta_lat/2), (row_lon-delta_lon/2),\
                    (row_lon+delta_lon/2)]
                    self._data_properties_dict["indexmap"].loc[last_index] = self._indexmap.loc[last_index]
                    self._data_properties_dict["dataset_df"] = self._dataset_df
                    self._num_entries += 1
                    self._data_properties_dict["num_entries"] = self._num_entries

        # update the types stored in the dataset dictionary
        self._data_properties_dict = set((list(rejected_types) + list(self._data_properties_dict["rejected_types"])))
        self._data_properties_dict["accepted_types"] = set(type_list + list(self._data_properties_dict["accepted_types"]))
        self._dataset_properties_dict["to_preproc_df"] = self._data_properties_dict["to_preproc_df"].append(data) # :D u remembered!

        return self._dataset_df


    def incr_date(self, date):
        '''
        Helper function that extends the daterange of self._dataset_df columns if necessary;
        modifies input self._dataset_df in place

        Input -
            date (DateTime)

        Outputs -
            (int) number of dates w/n updated DataFrame
        '''

        if isinstance(date, str):
            date = datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S').date()

        extended = False
        if date < self._data_properties_dict["min_date"]: # self._dataset_df.columns[4] needs to be this new date
            daterange = pd.date_range(date, self._data_properties_dict["min_date"])
            self._data_properties_dict["daterange"] = [x.date() for x in daterange.tolist()]
            front = self._dataset_df.columns.tolist()[0:5]
            back = self.dataset_df.columns.tolist()[6:]
            self._dataset_df = self._dataset_df.reindex(\
            columns=(front+self._data_properties_dict["daterange"]+back), fill_value=0)
            # prefix = np.zeros(shape=(self._dataset_df.shape[0], len(daterange)))
            # pre_df = pd.DataFrame(prefix, columns=list(daterange), \
            # index=self._dataset_df.index.tolist())
            self._data_properties_dict["min_date"] = date
            self.data_properties_dict["dataset_df"] = self._dataset_df
            extended = True
        elif date > self._data_properties_dict["max_date"]:
            daterange = pd.date_range(self._data_properties_dict["max_date"], date)
            self._data_properties_dict["daterange"] = [x.date() for x in daterange.tolist()]
            self._dataset_df = self._dataset_df.reindex(\
            columns=(self._dataset_df.columns.tolist()+self._data_properties_dict["daterange"][1:]), fill_value=0)
            self._data_properties_dict["max_date"] = date
            self.data_properties_dict["dataset_df"] = self._dataset_df
            extended = True
        # else: date w/n daterange
        if extended:
            return self.dataset_df.shape[1]-5 # first four columns are type, lat/lon start/stop
        else:
            return 0


    def merge(self, new_df, force_update=True):
        '''
        Merges two DataFrames of different event tyes
        set force_update to True (default) to update the dataproc's dataset properties dictionary
        with changes of the merge
        '''

        # assuming that they are the same of the same grid_size
        other_max_date = new_df.columns[-1]
        other_min_date = new_df.columns[4]

        # deepcopy the data_properties_dict and then reassign values of the proc
        if not force_update:
            prev_dataset_properties_dict = deepcopy(self._dataset_properties_dict)

        num_reps = self.incr_date(other_max_date)
        num_reps = self.incr_date(other_min_date)

        # two cases: one is if the new_df has larger grid_boxes than the current
        # other case is if the new_df has equal or smaller to the current df grid_boxes
        # want to loop through the smaller grid box sized df and add to the other df
        # but then there is the issue of the two not having the same lat/lon ranges
        # check for that and append the rest?
        # problem of having many cases, in the case of the smaller-sized grid exceeding
        # the lat/lon range of the bigger grid, could just append, but make sure to
        # check if dates need to be extended on the smaller sized-grid df

        # is this the kind of function he wanted(wrt. method or utility function)/
        # & are these considerations correct?


    def reverse_search(self, date, type_list, date_col="Date", type_col="Primary Type",\
                lat_col="Latitude", lon_col="Longitude", id_col="ID"):
        '''
        Returns selection of initially processed data
        Purpose is to verify the final combined dataframe was accurately created
        from the input data
        '''

        # proper way to search to confirm integrity of dataframe?
        # improper way would be to return values directly from the preproc_df

        if not isinstance(date, datetime.date):
            search_date = dparser.parse(date).date()
        types = set(type_list)

        res = self._dataset_df[self._dataset_df[search_date] == 1]
        res = res[res["Event_type"] == types]

        hit_ids = []
        for dataset_row in res.itertuples(index=True, name="Pandas"):
            lat_lb, lat_ub = dataset_row[2], dataset_row[3]
            lon_lb, lon_ub = dataset_row[4], dataset_row[5]

            for preproc_row in self._data_properties_dict["to_preproc_df"].itertuples(index=True, name="Pandas"):
                date_index = list(self._data_properties_dict["to_preproc_df"].columns).index(date_col)+1
                curr_date = preproc_row[date_index].date()
                lat_index = list(self._data_properties_dict["to_preproc_df"].columns).index(lat_col)+1
                curr_lat = preproc_row[lat_index]
                lon_index = list(self._data_properties_dict["to_preproc_df"].columns).index(lon_col)+1
                curr_lon = preproc_row[lon_index]
                type_index = list(self._data_properties_dict["to_preproc_df"].columns).index(type_col)+1
                event_type = str(preproc_row[type_index])
                id_index = list(self._data_properties_dict["to_preproc_df"].columns).index(id_col)+1
                id_val = int(preproc_row[id_index])

                # taking advantage of the fact that both df's are sorted by lat/lon
                if lat_ub >= curr_lat and lon_ub >= curr_lon:
                    if lat_lb <= curr_lat and lon_lb <= curr_lon and search_date == curr_date and\
                    event_type in types:
                        hit_ids.append(id_val)
                else:
                    break

        return self._data_properties_dict["to_preproc_df"].loc[self._data_properties_dict["to_preproc_df"][id_col].isin(hit_ids)]




    def cleanup(self):
        '''
        Maintenance function - deletes temporary directory containing files
        with dataproc data
        '''

        warnings.warn(\
        "WARNING: This spatiotemporaldataproc's data will be deleted and will be unusable henceforth.")
        shutil.rmtree(self._file_dir)
        self._file_dir = None



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
