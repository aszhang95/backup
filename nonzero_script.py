import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.patches import PathPatch
import numpy as np
import matplotlib.cm as cm
%matplotlib inline

def readTS(filename):
    df=pd.read_csv(filename,index_col=0)
    cz=[df.Event_type[i].replace('set([','')\
        .replace('])','').replace(' ','').replace("'",'')!='' for i in df.index]
    return df.loc[cz,:]

df1=readTS('100_test_df.csv')
print df1.shape
