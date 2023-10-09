import streamlit as st
import dataset
import time
import webbrowser
from pathlib import Path
# import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import streamlit.components.v1 as components
code1= '''
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
data = pd.read_csv("https://raw.githubusercontent.com/maulanamaib/topic_modelling/master/datavcm.csv")
del(data['Unnamed: 0'])
data '''
st.code(code1, language='python')
