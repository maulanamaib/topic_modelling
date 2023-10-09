import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
data = pd.read_csv("/content/drive/MyDrive/ppw/datavcm1.csv")
del(data['Unnamed: 0'])
data
