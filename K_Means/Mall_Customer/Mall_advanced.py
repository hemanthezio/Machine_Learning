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

#function to plot graph
def plot(X,cluster,centroid):
    clust=[]
    for i in cluster:
        clust.append(int(i))
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

#K-means++ initialization
def kpp(k,X):
    random.seed(1)
    cent=[]
    cent.append(X[random.randint(0,X.shape[0])].tolist())
    while(len(cent)<k):
        d2=[]
        for x in X:
            minn=[]
            for c in np.array(cent):
                minn.append(distance.minkowski(x,c,2))
            d2.append(min(minn))
        d2=np.array(d2).reshape(X.shape[0],1)
        prob=d2/d2.sum()
        cprob=prob.cumsum() 
        r=random.random()
        idx=np.where(cprob>=r)[0][0]
        cent.append(X[idx].tolist())
    return np.array(cent)


#Reinitalize means
def reinit(k,cluster,data_arr,centroid):
    summ=np.zeros((k,len(data_arr[0])))
    count=[]
    
    for p in range(k):
        index = np.where(cluster==p)[0]
        summ[p]=(np.sum(X[index],axis=0,keepdims=True))
        count.append(index.shape[0])
    
    summ=np.array(summ)
        
    for q in range(k):
        if count[q]>0 :
            centroid[q]=summ[q]/count[q]
    
    return centroid

#Kmeans algorithm
def kmeans(k,max_itr,x,init_method,distance_measure,power=3):
    cluster=[]
    wcss=[]
    centroid=[]
    old_centroid=[]
    n=X.shape[0]
    index=0
    
    #choose initi method
    if(init_method=='random'):
        random.seed(3)
        for i in range(k):
            centroid.append(X[random.randint(1,n)])
        centroid=np.array(centroid)
    elif(init_method=='K++'):
        centroid=kpp(k,x)
    else:
        centroid=X[list(range(k)),:]
    print("\nInitialization method :"+init_method)
    
    #choose distance measure
    if(distance_measure=='Manhattan'):
        h=1
    elif(distance_measure=='Euclidean'):
        h=2
    elif(distance_measure=='Minkowski'):
        h=power
    print("Distance measure :"+distance_measure)
        
    for i in range(n):
        cluster.append(-1)
    cluster=np.array(cluster).reshape(n,1)
    
    print("K=",k)
    #kmeans logic
    for i in range(max_itr):
        old_centroid=np.copy(centroid)                
        for j in range(n):
            mini=99999
            for z in range(k):
                tmp=distance.minkowski(x[j],centroid[z],h)
                if tmp<mini :
                    mini=tmp
                    index=z
            cluster[j]=index
        
        centroid=reinit(k,np.array(cluster),x,centroid)
        
        #if centroid donot change convergence point!
        if(old_centroid==centroid).all():
            print("\nConverged at iteration",i,"\n")
            break
        
    #calculate wcss
    total=0
    for r in range(n):
        total=total+distance.minkowski(x[r],centroid[cluster[r]],h) 
    wcss.append([total,k])
    return cluster,centroid,wcss

# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values
x=dataset.iloc[:, [3]].values
y=dataset.iloc[:, [4]].values
data_arr=np.array(X)

#Kmeans for arbitary range of k
print("\n\nStating elbow analysis.......")
cluster=[]
centroid=[]
wcss_m=[]
for k in range(1,10):
    cluster,centroid,wcss=kmeans(k,50,X,init_method="k++",distance_measure="Euclidean")
    wcss_m.append(wcss[0])

#Elbow method analysis to find optimal k
print("######################       Elbow analysis       ##############################")
wcss_arr=np.array(wcss_m)
plt.plot(wcss_arr[:,[1]],wcss_arr[:,[0]])
plt.title('Elbow curve analysis')
plt.ylabel('WCSS')
plt.xlabel('K')
plt.show()


#use optimal k
k=5
print("\n\n Optimal K value from Elbow analysis:",k)

#use different init method
print("\n######################   use different init method   ##############################")
init_method=["K++","random","1st k"]
for init in init_method:
    k=5
    cluster,centroid,wcss=kmeans(k,100,X,init_method=init,distance_measure="Euclidean")
    count=[]
    for p in range(k):
            index = np.where(cluster==p)[0]
            count.append(index.shape[0])
        
    print("centroids :\n",centroid)
    print("cluster member count :\n",count)
    print("WCSS:\n",wcss)
    plot(X,cluster,centroid)
    print("---------------------------------------------------------------")

#use different distance measure
print("#######################    use different distance measure   #######################")
distance_measure=["Manhattan","Euclidean","Minkowski"]
for dist in distance_measure:
    k=5
    cluster,centroid,wcss=kmeans(k,100,X,init_method="k++",distance_measure=dist)
    count=[]
    for p in range(k):
        index = np.where(cluster==p)[0]
        count.append(index.shape[0])
        
    print("centroids :\n",centroid)
    print("cluster member count :\n",count)
    print("WCSS :\n",wcss)
    plot(X,cluster,centroid)
    print("---------------------------------------------------------------")


    




        


