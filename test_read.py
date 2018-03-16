#!/usr/bin/python


import pandas as pd
import numpy as np
import time
import pickle
from datetime import datetime
from datetime import timedelta
#from joblib import Parallel, delayed
import os
import sys
from tqdm import tqdm


NOPICKLE=False
if NOPICKLE:
    df=pd.read_csv('Crimes2001-2017.csv')
    df.Date=pd.to_datetime(df.Date)
    df.to_pickle('alldata.p')
else:
    df=pd.read_pickle('alldata.p')

NUM=1
num=0
LEN=5620

def getTS(df,crime_types=['HOMICIDE','BATTERY','ASSAULT'],tile=[41.8,41.81,-87.75,-87.74],
          INIT='1/1/2001',LEN=LEN,FREQ='D',Cols=['Primary Type','Latitude','Longitude','Date']):

    TS_NAME="#".join(str(x) for x in tile)

    lat_=tile[0:2]
    lon_=tile[2:4]

    df=df[Cols].loc[df['Primary Type'].isin(crime_types)].sort_values(by='Date').dropna()
    df=df.loc[(df['Latitude'] > lat_[0])
         & (df['Latitude'] <= lat_[1])
         & (df['Longitude'] > lon_[0])
         & (df['Longitude'] <= lon_[1])]
    df.index = df.Date
    #  Now have dataframe with index date sorted in order that have the correct crime types
    # and are within the square specified

    df=df[['Primary Type']].isin(crime_types)+0.0 # turning into 0's and 1's
    trng_ = pd.date_range(INIT, periods=LEN, freq=FREQ)
    # looping through the daterange and getting that date, i, then looking that up
    # in the df
    ts=[df.loc[trng_[i]:trng_[i+1]].size for i in np.arange(trng_.size-1)]

    return pd.DataFrame(ts,columns=[TS_NAME],index=trng_[:-1]).transpose()

CSVfile='TS.csv'
EPS=0.005
THRESHOLD=0.2
total = len(sys.argv)
if total > 1:
    EPS = float(sys.argv[1])
if total > 2:
    LAT1 = float(sys.argv[2])
if total > 3:
    LAT2 = float(sys.argv[3])
if total > 4:
    LON1 = float(sys.argv[4])
if total > 5:
    LON2 = float(sys.argv[5])
if total > 6:
    CSVfile = sys.argv[6]
if total > 7:
    THRESHOLD = float(sys.argv[7])

LAT=np.arange(LAT1,LAT2,EPS)
LON=np.arange(LON1,LON2,EPS)


TS=pd.concat([getTS(df,crime_types=['HOMICIDE','BATTERY','ASSAULT'],tile=[i,i+EPS,j,j+EPS]) for i in tqdm(LAT) for j in LON])

statbool=TS.astype(bool).sum(axis=1)/LEN
statbool.to_csv('statbool.dat')
TS.loc[statbool > THRESHOLD].to_csv(CSVfile,sep=" ")
