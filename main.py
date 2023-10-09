import streamlit as st
import time
import webbrowser
from pathlib import Path
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import streamlit.components.v1 as components
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
code1= '''
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
data = pd.read_csv("https://raw.githubusercontent.com/maulanamaib/topic_modelling/master/datavcm.csv")
del(data['Unnamed: 0'])
data '''

code2 ='''
lda = LatentDirichletAllocation(n_components=3, doc_topic_prior=0.2, topic_word_prior=0.1,random_state=42,max_iter=1)
lda_top=lda.fit_transform(data)
print(lda_top.shape)
print(lda_top)
'''
st.code(code1, language='python')

data = pd.read_csv("https://raw.githubusercontent.com/maulanamaib/topic_modelling/master/datavcm.csv")
del(data['Unnamed: 0'])
data

st.code(code2, language='python')

lda = LatentDirichletAllocation(n_components=3, doc_topic_prior=0.2, topic_word_prior=0.1,random_state=42,max_iter=1)
lda_top=lda.fit_transform(data)
print(lda_top.shape)
print(lda_top)
