import numpy as np
import pandas as pd
import datetime
from dateutil.parser import parse
import csv
import dateutil.parser as dparser

min_lat = 36.619446395
max_lat = 42.022644813

min_lon = -91.686565684
max_lon = -87.524529378

grid_size = 250

min_date = datetime.date(2001, 1, 1)
max_date = datetime.date(2018, 1, 1)

dlat = (max_lat - min_lat)/grid_size
dlon = (max_lon - min_lon)/grid_size
lat_pres = 9
lon_pres = 9


lat_start = []
lat_stop = []
lon_start = []
lon_stop = []

curr_lat = min_lat
curr_lon = min_lon

# the problem with this is that you're only creating the diagonal coordinates
for i in range(grid_size):
    for j in range(grid_size):
            lat_start.append(curr_lat)
            lat_stop.append(curr_lat+(dlat-float("0."+"0"*(lat_pres-1)+"1")))
            lon_start.append(curr_lon)
            lon_stop.append(curr_lon+(dlon-float("0."+"0"*(lon_pres-1)+"1")))
            curr_lon += dlon
    curr_lat += dlat
    curr_lon = min_lon

type_col_ = [set() for _ in xrange(len(lat_start))]

grid_df = pd.DataFrame.from_dict({"Event_type": type_col_, "Latitude_start":lat_start, "Latitude_stop":lat_stop, \
"Longitude_start":lon_start, "Longitude_stop":lon_stop})
grid_df.index.name = "grid_id"

daterange = pd.date_range(min_date, max_date)
date_range = [x.date() for x in daterange.tolist()]
zeros = np.zeros(shape=(len(lat_start), len(date_range)))
ts_df = pd.DataFrame(zeros, columns=date_range, dtype=int)

df = pd.concat([grid_df, ts_df], axis=1)
