# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 22:00:30 2018

@author: Hemanth kumar
"""
''' using K-Means Clustering to segment clients into different groups 
 based on spending and annual income  using the mall dataset '''

# K-Means Clustering
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from scipy.spatial import distance

#Reinitalize means
def reinit(k,cluster,data_arr,centroid):
    summ=np.zeros((k,len(data_arr[0])))
    count=[]
    
    for p in range(k):
        index = np.where(cluster==p)[0]
        index=index.reshape(len(index),1)
        
        summ[p]=(np.sum(X[index],axis=0,keepdims=True))
        count.append(index.shape[0])
    
    summ=np.array(summ)
        
    for q in range(k):
        if count[q]>0 :
            centroid[q]=summ[q]/count[q]
    
    return centroid

#Kmeans algorithm
def kmeans(k,max_itr,x):
    '''k=3
    max_itr=50
    x=X'''
    cluster=[]
    wcss=[]
    centroid=[]
    old_centroid=[]
    n=X.shape[0]
    index=0
    np.random.seed(1)
    for i in range(k):
        centroid.append(random.choice(X))
    centroid=np.array(centroid)
    for i in range(n):
        cluster.append(-1)
    cluster=np.array(cluster).reshape(n,1)
    
            
    for i in range(max_itr):
        #print(centroid)
        old_centroid=np.copy(centroid)                
        for j in range(n):
            mini=99999
            for z in range(k):
                tmp=distance.euclidean(x[j],centroid[z])
                if tmp<mini :
                    mini=tmp
                    index=z
            cluster[j]=index
            centroid=reinit(k,np.array(cluster),x,centroid)
            
        if(old_centroid==centroid).all():
            print("old",old_centroid)
            print("new",centroid)
            print("Converged at itr",i)
            break
        
    total=0
    for r in range(n):
        total=total+distance.euclidean(x[r],centroid[cluster[r]]) 
    wcss.append([total,k])
    return cluster,centroid,wcss


# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values
x=dataset.iloc[:, [3]].values
y=dataset.iloc[:, [4]].values
data_arr=np.array(X)


#Kmeans for arbitary range of k
cluster=[]
centroid=[]
wcss_m=[]
for k in range(1,10):
    cluster,centroid,wcss=kmeans(k,50,X)
    wcss_m.append(wcss[0])

#Elbow method analysis to find optimal k
wcss_arr=np.array(wcss_m)
plt.plot(wcss_arr[:,[1]],wcss_arr[:,[0]])
plt.title('Elbow curve analysis')
plt.ylabel('WCSS')
plt.xlabel('K')
plt.show()

#use optimal k
k=5
cluster,centroid,wcss=kmeans(k,3000,X)
count=[]
for p in range(3):
        index = np.where(cluster==p)[0]
        count.append(index.shape[0])
        
print("centroids:",centroid)
print("cluster member count:",count)        
clust=[]
for i in cluster:
    clust.append(int(i))
#visualization
# Visualising the clusters
#plt.scatter(x,y)
plt.scatter(X[np.array(clust) == 0, 0], X[np.array(clust) == 0, 1], s = 50, c = 'red', label = 'Careful')
plt.scatter(X[np.array(clust) == 1, 0], X[np.array(clust) == 1, 1], s = 50, c = 'blue', label = 'Standard')
plt.scatter(X[np.array(clust) == 2, 0], X[np.array(clust) == 2, 1], s = 50, c = 'green', label = 'Target')
plt.scatter(X[np.array(clust) == 3, 0], X[np.array(clust) == 3, 1], s = 50, c = 'cyan', label = 'Careless')
plt.scatter(X[np.array(clust) == 4, 0], X[np.array(clust) == 4, 1], s = 50, c = 'magenta', label = 'Sensible')
plt.scatter(centroid[:, 0],centroid[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()