# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 17:09:06 2020

@author: HP
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

airlines=pd.read_excel('C:/Users/HP/Desktop/assignments submission/clustering/EastWestAirlines.xlsx',sheet_name='data')
airlines.isna().sum()
def norm_func(i):
    x=(i-i.mean())/i.std()
    return x

norm_data=norm_func(airlines.iloc[:,1:])

############screw plot##########
k=list(range(2,15))
k
TWSS=[]
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
for i in k:
    kmean=KMeans(n_clusters=i)
    kmean.fit(norm_data)
    WSS=[]
    for j in range(i):
        WSS.append(sum(cdist(norm_data.iloc[kmean.labels_==j,:],kmean.cluster_centers_[j].reshape(1,norm_data.shape[1]),metric='euclidean')))
    TWSS.append(sum(WSS))

plt.plot(k,TWSS,'ro-');plt.xlabel('k values');plt.ylabel('TWSS')

############k-means clustering#########

#k=8 from graph
model=KMeans(n_clusters=8).fit(norm_data)
clusters=pd.DataFrame(model.labels_)
# Noisy samples have label as -1
clusters['clusters']=clusters[0]
clusters.drop(columns=0,inplace=True)
clusters.value_counts()
final_data=pd.concat([clusters,airlines],axis=1)

insights=final_data.groupby(final_data.clusters).mean()

final_data.to_csv('airlines_k_final.csv')
