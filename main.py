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

code3 ='''
U = pd.DataFrame(lda_top, columns=['Topik 1','Topik 2','Topik 3'])
U'''

code4 = '''
print(lda.components_)
print(lda.components_.shape)'''

code5 ='''
label=[]
for i in range (1,(lda.components_.shape[1]+1)):
  masukan = data.columns[i-1]
  label.append(masukan)
VT_tabel = pd.DataFrame(lda.components_,columns=label)
VT_tabel.rename(index={0:"Topik 1",1:"Topik 2",2:"Topik 3"}).transpose()'''


st.code(code1, language='python')

data = pd.read_csv("https://raw.githubusercontent.com/maulanamaib/topic_modelling/master/datavcm.csv")
del(data['Unnamed: 0'])
data

st.code(code2, language='python')

lda = LatentDirichletAllocation(n_components=3, doc_topic_prior=0.2, topic_word_prior=0.1,random_state=42,max_iter=1)
lda_top=lda.fit_transform(data)
print(lda_top.shape)
print(lda_top)

st.code(code3, language='python')
U = pd.DataFrame(lda_top, columns=['Topik 1','Topik 2','Topik 3'])
U
st.code(code4, language='python')
print(lda.components_)
print(lda.components_.shape)
st.code(code5, language='python')
label=[]
for i in range (1,(lda.components_.shape[1]+1)):
  masukan = data.columns[i-1]
  label.append(masukan)
VT_tabel = pd.DataFrame(lda.components_,columns=label)
VT_tabel.rename(index={0:"Topik 1",1:"Topik 2",2:"Topik 3"}).transpose()
