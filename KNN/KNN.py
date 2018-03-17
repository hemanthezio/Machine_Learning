# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 21:37:22 2018

@author: Hemanth kumar
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

#imporint the dataset
from sklearn.datasets import load_wine
dataset = load_wine()
classes=dataset.target[[10, 80, 140]].tolist()

X=dataset.data
y=dataset.target
y=y.reshape(len(y),1)

#Feature Scaling
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.11, random_state = 0)

#Train
def train(X_train, y_train):
    #do nothing
    return

#To return most common class 
def most_common(lst):
    data = Counter(lst)
    return data.most_common(1)[0][0]

#Predict the class among K-Nearest neighbours
def predict(x_train,y_train,x_test,k):
    distance=[]
    knn=[]
    
    for i in range(len(x_train)):
        dist=(np.sqrt(np.sum(np.square(x_train[i,:]-x_test))))
        distance.append([dist,i])
    
    distance=sorted(distance)
    
    for i in range(k):
        index=distance[i][1]
        knn.append(int(y_train[index]))
    
    return most_common(knn)

#Driver function
def KNN(x_train, y_train, x_test,  k):
	
    if k > len(X_train):
        raise ValueError
    
    # train on the input data
    train(X_train, y_train)

    y_pred=[]
    
	 # loop over all observations
    for i in range(len(X_test)):
        y_pred.append(predict(x_train, y_train, x_test[i, :].reshape(1,len(x_test[0])), k))
    return y_pred



#Train set accuracy
y_pred=[]
try:
    y_pred=KNN(X_train, y_train, X_test, 7)
    y_pred = np.asarray(y_pred)

    # evaluating accuracy
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(y_test, y_pred) * 100
    print('\nTrain accuracy of classifier is ',accuracy)

except ValueError:
    print('Can\'t have more neighbors than training samples!!')


#Test set accuracy
y_pred=[]
try:
    y_pred=KNN(X_train, y_train, X_test, 21)
    y_pred = np.asarray(y_pred)

    # evaluating accuracy
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(y_test, y_pred) * 100
    print('\nTest accuracy of  classifier is ',accuracy)

except ValueError:
    print('Can\'t have more neighbors than training samples!!')
    