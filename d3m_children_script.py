# assume running from within pycode

from cw7 import *
from d3m_embedding_child import *

print("Running d3m_embedding_child with num_dimensions=2")
bin_path = os.path.abspath("../bin")

print("Reading in data from deploy_scripts/examples/data.dat and setting dtype=np.int32")
test = SupervisedModelBase(bin_path)
X_df = test.read_in_ragged("../deploy_scripts/examples/data.dat", " ")
X = X_df.values
X_ = X.astype(np.int32)

def q(x):
    if x <= 0:
        return 0
    else:
        return 1

print("Applying np.vectorize onto quantizer")
vq = np.vectorize(q)

data_class = Input(data=X_, is_categorical=True, is_synchronized=True, preproc=vq)
sec = SmashEmbedding(bin_path, data_class, 2)

print("Calling attributes of SmashEmbedding class to ensure proper configuration: problem_type = {}, \
num_dimensions = {}, MDS = {}".format(sec.problem_type, sec.num_dimensions, sec.primitive))
print("Running sec.fit(nr=3, d=True)")
res = sec.fit(nr=3, d=True)
print(res)

print("Running sec.fit_transform(nr=3, d=True)")
res2 = sec.fit_transform(nr=3, d=True)
print(res2)

print("Running sec.sklearn_MDS_embed(nr=2, d=True)")
res3 = sec.sklearn_MDS_embed(nr=2, d=True)
print(res3)
