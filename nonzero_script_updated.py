import numpy as np
import pandas as pd

def readTS(filename):
    df=pd.read_csv(filename,index_col=0)
    cz=[df.Event_type[i].replace('set([','')\
        .replace('])','').replace(' ','').replace("'",'')!='' for i in df.index]
    return df.loc[cz,:]

df1=readTS('../../../../../../project2/ishanu/CRIME/data/TS1_200.csv')
print df1.shape
