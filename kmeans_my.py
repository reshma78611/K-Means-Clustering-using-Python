# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 22:38:02 2020

@author: HP
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

univ=pd.read_csv('C:/Users/HP/Desktop/python prgrmg/clustering/Universities.csv')
univ.isna().sum()
#normalize the data
def norm_func(i):
    x=(i-i.mean())/i.std()
    return x
norm_data=norm_func(univ.iloc[:,1:])
norm_data.head()

#from sklearn.preprocessing import scale
#norm_data=pd.DataFrame(scale(univ.iloc[:,1:]))
#norm_data.head()


############screw plot or elbow curve##########
k=list(range(2,15))
k
TWSS=[]
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
for i in k:
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(norm_data)
    WSS=[]
    for j in range(i):
        WSS.append(sum(cdist(norm_data.iloc[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1,norm_data.shape[1]),metric="euclidean")))
    TWSS.append(sum(WSS))

#screw plot
plt.plot(k,TWSS,'ro--');plt.xlabel('no of clusters');plt.ylabel('TWSS')#;plt.xticks(k)

# Selecting 5 clusters from the above scree plot which is the optimum number of clusters 
model=KMeans(n_clusters=5).fit(norm_data)
cluster_labels=pd.Series(model.labels_)
cluster_labels.value_counts()
final_data=pd.concat([cluster_labels,univ],axis=1)
final_data.rename(columns={0:'clusters'},inplace=True)

final_data.iloc[:,1:7].groupby(final_data.clusters).mean()
final_data.to_csv('final_k_univ.csv',index=False)
