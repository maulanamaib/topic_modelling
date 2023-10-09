import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
data = pd.read_csv("https://raw.githubusercontent.com/maulanamaib/topic_modelling/master/datavcm.csv")
del(data['Unnamed: 0'])
data
