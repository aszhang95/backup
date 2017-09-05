import d3m_unsups
from d3m_unsups import  *
import numpy as np


print d3m_unsups.__version__

input=Input()


def func(d):
    return d

data=np.array([[0,1,1,0,1],[1,0,0,0,1]],dtype='float64')


print data.dtype

input=Input(data,is_symbolic=True)

print input.data
print input.is_symbolic


input.data=data
input.transform = func

print input.data
print input.is_symbolic

print input.get()


output=Output()
output.model={(3,):'str'}
output.prediction={(2,1):input}


print output.model
print output.prediction
print input
print output


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


    
ds=data_smashing('distance_metric_learning')
ds.data=Input(data,is_synchronized=True,transform=func)


print "DS DATA" ,ds.data, ds.output

ds.output=output
ds.set_param(**{'eps':0.2})
print ds.output, ds.problem_type

print ds._hyperparameters

ds.performance()
