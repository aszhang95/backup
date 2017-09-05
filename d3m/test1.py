import d3m_unsups
from d3m_unsups import  *
import numpy as np
import pandas as pd

print d3m_unsups.__version__


def qtz(ar_):
    b=[]
    for ai in ar_:
        if ai <= 0:
            b.append(0)
        else:
            b.append(1)
    return np.array(b,dtype=object)

def preproc(d):
    return np.apply_along_axis(qtz, 0, d)


class data_smashing(Unsupervised_Series_Learning_Base):

    def fit(self,*arg,**kwds):
        return 2.0

    def fit_transform(self,*arg,**kwds):
        pass

    def predict(self,*arg,**kwds):
        pass

    def predict_proba(self,*arg,**kwds):
        pass

    def log_proba(self,*arg,**kwds):
        pass

    def performance(self,*arg,**kwds):
        pass

    def score(self,*arg,**kwds):
        pass
    def transform(self,*arg,**kwds):
        pass
    def fit_predict(self,*arg,**kwds):
        pass



data=pd.read_csv('data.dat',sep=" ",header=None).values

ds=data_smashing('distance_metric_learning')
ds.data=Input(data,is_synchronized=True,preproc=preproc)

print "DS DATA" ,ds.data, ds.output
#ds.output=output
ds.set_param(**{'eps':0.2})
print ds.output, ds.problem_type
print ds._hyperparameters
print ds.data.get(), type(ds.data.get())
ds.performance()

