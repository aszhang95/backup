import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from datetime import timedelta
from tqdm import tqdm, tqdm_pandas
from haversine import haversine
from sodapy import Socrata

"""
Utilities for spatio temporal analysis
@author zed.uchicago.edu
"""

__version__='0.31415'

class spatioTemporal:
    """
    Utilities for spatio temporal analysis
    @author zed.uchicago.edu

    Crime Prediction implementation of spatio temporal data class; reads in spatio temporal data in a
    format easily ingested by XgenESeSS binaries

    Attributes:
        log_store (Pickle): Pickle storage of class data & dataframes
        log_file (string): path to CSV of legacy dataframe
        ts_store (string): path to CSV containing most recent ts export
        DATE (string):
        EVENT (string): column label for category filter
        coord1 (string): first coordinate level type; is column name
        coord2 (string): second coordinate level type; is column name
        coord3 (string): third coordinate level type; is column name (z coordinate)
        end_date (datetime.date): upper bound of daterange for timeseries analysis
        freq (string): timeseries increments; e.g. D for date
        columns (list): list of column names to use;
            required at least 2 coordinates and event type
        types (list of strings): event type list of filters
        value_limits (tuple): boundaries (magnitude of the event for earthquake; magnitudes above threshold)
        grid (pd.DataFrame): dataframe containing location tile and corrresponding timeseries
        threshold (float): significance threshold
    """

    def __init__(self,
            log_store='log.p',
            log_file=None,
            ts_store=None,
            DATE='Date',
            EVENT='Primary Type',
            coord1='Latitude',
            coord2='Longitude',
            coord3=None,
            init_date=None,
            end_date=None,
            freq=None,
            columns=None,
            types=None,
            value_limits=None,
            grid=None,
            threshold=None):

        assert not ((types is not None)
                    and (value_limits is not None))

        if log_file is not None:
            df = pd.read_csv(log_file)
            df[DATE] = pd.to_datetime(df[DATE])
            df.to_pickle(log_store)
        else:
            df = pd.read_pickle(log_store)

        self._logdf = df
        self._spatial_tiles = None
        self._dates = None
        self._THRESHOLD=threshold
        self._livepath =livepath

        if freq is None:
            self._FREQ = 'D'
        else:
            self._FREQ=freq

        self._DATE = DATE

        if init_date is None:
            self._INIT = '1/1/2001'
        else:
            self._INIT = init_date

        if end_date is not None:
            self._END = end_date
        else:
            self._END=None

        self._EVENT = EVENT
        self._coord1 = coord1
        self._coord2 = coord2
        self._coord3 = coord3

        if columns is None:
            self._columns = [EVENT, coord1, coord2, DATE]
        else:
            self._columns = columns

        self._types=types
        # coordinate limits?
        self._value_limits=value_limits

        self._ts_dict = {}

        self._grid={}
        if grid is not None:
            assert(self._coord1 in grid)
            assert(self._coord2 in grid)
            assert('Eps' in grid)

            self._grid[self._coord1]=grid[self._coord1]
            self._grid[self._coord2]=grid[self._coord2]
            self._grid['Eps']=grid['Eps']

        self._trng = pd.date_range(start=self._INIT,
                                   end=self._END,freq=self._FREQ)


    def getTS(self,_types=None,tile=None):
        """
        Utilities for spatio temporal analysis
        @author zed.uchicago.edu

        Given location tile boundaries and type category filter, creates the
        corresponding timeseries as a pandas DataFrames

        Inputs:
            _types (list of strings): list of category filters
            tile (list of floats): location boundaries for tile

        Outputs:
            pd.Dataframe of timeseries data to corresponding grid tile
        """

        assert(self._END is not None)
        TS_NAME = ('#'.join(str(x) for x in tile))+"#"+stringify(_types)

        lat_ = tile[0:2]
        lon_ = tile[2:4]

        if self._value_limits is None:
            df = self._logdf[self._columns]\
                     .loc[self._logdf[self._EVENT].isin(_types)]\
                     .sort_values(by=self._DATE).dropna()
        else:
            df = self._logdf[self._columns]\
                     .loc[self._logdf[self._EVENT]\
                          .between(self._value_limits[0],
                                   self._value_limits[1])]\
                     .sort_values(by=self._DATE).dropna()

        df = df.loc[(df[self._coord1] > lat_[0])
                    & (df[self._coord1] <= lat_[1])
                    & (df[self._coord2] > lon_[0])
                    & (df[self._coord2] <= lon_[1])]
        df.index = df[self._DATE]
        df=df[[self._EVENT]]

        ts = [df.loc[self._trng[i]:self._trng[i + 1]].size for i in
              np.arange(self._trng.size - 1)]

        return pd.DataFrame(ts, columns=[TS_NAME],
                            index=self._trng[:-1]).transpose()


    def timeseries(self,LAT,LON,EPS,_types,CSVfile='TS.csv',THRESHOLD=None):
        """
        Utilities for spatio temporal analysis
        @author zed.uchicago.edu

        Creates grid of location tiles and their respective timeseries from
        input datasource with
            significance threshold THRESHOLD
            latitude, longitude coordinate boundaries given by LAT, LON

        Input:
            LAT (float):
            LON (float):
            EPS (float): coordinate increment ESP
            _types (list): event type filter; accepted event type list
            CSVfile (string): path to output file

        Output:
            (None): grid pd.Dataframe written out as CSV file to path specified
        """

        if THRESHOLD is None:
            if self._THRESHOLD is None:
                THRESHOLD=0.1
            else:
                THRESHOLD=self._THRESHOLD

        if self._trng is None:
            self._trng = pd.date_range(start=self._INIT,
                                       end=self._END,freq=self._FREQ)

        _TS = pd.concat([self.getTS(tile=[i, i + EPS, j, j + EPS],
                                    _types=_types) for i in tqdm(LAT)
                         for j in tqdm(LON)])

        LEN=pd.date_range(start=self._INIT,
                          end=self._END,freq=self._FREQ).size+0.0

        statbool = _TS.astype(bool).sum(axis=1) / LEN
        _TS = _TS.loc[statbool > THRESHOLD]
        self._ts_dict[repr(_types)] = _TS

        if CSVfile is not None:
            _TS.to_csv(CSVfile, sep=' ')

        return


    def fit(self,grid=None,INIT=None,END=None,THRESHOLD=None):
        """
        Utilities for spatio temporal analysis
        @author zed.uchicago.edu

        Rebuild grid from new input arguments (?)

        Inputs:
            grid (pd.DataFrame): dataframe of location timeseries data
            INIT (datetime.date): starting timeseries date
            END (datetime.date): ending timeseries date
            THRESHOLD (float): significance threshold

        Outputs:
            (None)
        """

        if INIT is not None:
            self._INIT=INIT
        if END is not None:
            self._END=END
        if grid is not None:
            self._grid=grid

        assert(self._END is not None)
        assert(self._coord1 in self._grid)
        assert(self._coord2 in self._grid)
        assert('Eps' in self._grid)

        if self._types is not None:
            for key in self._types:
                self.timeseries(self._grid[self._coord1],
                                self._grid[self._coord2],
                                self._grid['Eps'],
                                key,
                                CSVfile=stringify(key)+'.csv',
                                THRESHOLD=THRESHOLD)
            return
        else:
            assert(self._value_limits is not None)
            self.timeseries(self._grid[self._coord1],
                            self._grid[self._coord2],
                            self._grid['Eps'],
                            None,
                            CSVfile=stringify(key)+'.csv',
                            THRESHOLD=THRESHOLD)
            return


    def pull(self):
        """
        Utilities for spatio temporal analysis
        @author zed.uchicago.edu

        Pulls new entries from datasource
        NOTE: should make flexible but for now use city of Chicago data
        """

        socrata_domain = "data.cityofchicago.org"
        socrata_dataset_identifier = "crimes"
        socrata_token = "ZIgqoPrBu0rsvhRr7WfjyPOzW"

        client = Socrata(socrata_domain, socrata_token)


    def readTS():
        """
        Utilities for spatio temporal analysis
        @author zed.uchicago.edu
        """

        pass


    def generateNeighborMap(self):
        """
        Utilities for spatio temporal analysis
        @author zed.uchicago.edu

        For pair-wise model generation; pick only the nearest neighbors to track of a
        point and create models of it

        Input:
            None
        Output:
            (list): the mappings
        """

        A=[]
        for key,value in self._ts_dict.iteritems():
            A.append([i.replace("#"," ").split()[0:4] for i in value.index])
        print A


def stringify(List):
    """
    Utility function
    @author zed.uchicago.edu

    Converts list into string separated by dashes or empty string if input list
    is not list or is empty

    Input:
        List (list): input list to be converted

    Output:
        (string)
    """

    if List is None:
        return ''
    if not List:
        return ''

    return '-'.join(str(elem) for elem in List)
