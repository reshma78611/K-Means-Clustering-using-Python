# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 16:31:17 2020

@author: HP
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

crime=pd.read_csv('C:/Users/HP/Desktop/assignments submission/clustering/crime_data.csv')
crime.isna().sum()

def norm_func(i):
    x=(i-i.mean())/i.std()
    return x
norm_data=norm_func(crime.iloc[:,1:])

#################screw plot/elbow graph###########
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
        WSS.append(sum(cdist(norm_data.iloc[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1,norm_data.shape[1]),metric='euclidean')))
    TWSS.append(sum(WSS))

plt.plot(k,TWSS,'ro-');plt.xlabel('k values');plt.ylabel('TWSS')

##########k-means clustering############

#select k=6 as optimum clusters
model=KMeans(n_clusters=6).fit(norm_data)
cluster_labels=pd.DataFrame(model.labels_)
cluster_labels['clusters']=cluster_labels[0]
final_data=pd.concat([cluster_labels,crime],axis=1)

insights=final_data.groupby(final_data.clusters).mean()

final_data.to_csv('k_crime_final.csv')
