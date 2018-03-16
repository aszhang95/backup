#!/usr/bin/python

import spZED as sp
import pandas as pd
import numpy as np



ts=sp.readTS('TS.csv',csvNAME='TS1',
             BEG='2010-01-01',END='2015-12-31')

#ts.to_csv('TS1.csv',sep=" ")

sp.splitTS('TS.csv',csvNAME='TS1',
           BEG='2010-01-01',END='2017-01-01',
           dirname='./data/',prefix="@")
