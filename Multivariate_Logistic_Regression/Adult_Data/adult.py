# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 21:02:39 2018

@author: Hemanth kumar
"""
#import required library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from progressbar import ProgressBar
from sklearn.metrics import accuracy_score
#Import dataset
dataset = pd.read_csv('adult.csv')
dataset_clean=dataset.copy()


#Replace missing values "?" with np.nan
dataset_clean=dataset_clean.replace(to_replace="?",value=np.nan)

#check for missing values
dataset_clean.isnull().sum()

#Handling missing values
null=dataset_clean.isnull().sum()
null_index=np.where(np.array(null)>0)
null=null.index[null_index]

num_cols=list(dataset._get_numeric_data().columns)
cat_col=set(dataset.columns)-set(dataset._get_numeric_data().columns)
cat_col=list(cat_col)
columns=list(dataset.columns)

for col in columns:
    if(col in null):
        if(col in num_cols):
            dataset_clean.fillna(dataset_clean.mean()[col])
        else:
            dataset_clean[col]=dataset_clean[col].fillna(dataset_clean[col].mode()[0])
        
dataset_clean.isnull().sum()

#Categorial encoding
from sklearn.preprocessing import LabelEncoder
for col in cat_col:
    labelencoder_X1= LabelEncoder()
    dataset_clean[col]=labelencoder_X1.fit_transform(dataset_clean[col])
    

dataset_clean.describe()

#Seperating the variables
X=dataset_clean.iloc[:,:-1].values
y=dataset_clean.iloc[:,[-1]].values

dataset_clean.corr()

# Feature Scaling using standard scalar
def featureScale(X):
    means=np.mean(X,axis=0)
    stdev=np.std(X,axis=0)
    X = (X - means)/stdev
    return X
X=featureScale(X)

#append 1 for theta0
X=np.append(arr=np.ones((len(X),1)).astype(int),values=X,axis=1)


# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10, random_state = 0)

#Hypothesis
def h(x,theta):
    thetax=np.dot(theta.T,x)
    hx=1/(1+np.exp(-thetax))
    return hx

#Gradient descent
def gradient_descent(alpha,x,y,max_itr):
    pbar = ProgressBar()
    m =len(x)
    n=len(x[0])
    J_log=list()
    
    np.random.seed(3)
    # initial theta
    theta=list()
    for i in range(n):
        theta.append([random.random()])
    theta=np.array(theta)
    
    
    # total error, J(theta)
    yzero=0
    yone=0
    for i in range(m):
        yone+=y[i]*np.log(h(x[i],theta))
        yzero+=(1-y[i])*np.log(1-h(x[i],theta))
    J=yone+yzero
    J=float(J*(-1/m))
    J_log.append(J)
    print('\ninitial error=',float(J) )
    
    for i in pbar(range(max_itr)):
        hx=list()
        for j in range(m):
            hx.append((h(x[j],theta)))
        hx=np.array(hx)
        hx=hx.reshape(m,1)
        err=np.array(hx)-y
        err=err*x
        summ=np.sum(err,axis=0)
        summ=summ.reshape(n,1)
        grad=alpha*summ
        theta=theta-grad
        
        # error
        yzero=0
        yone=0
        for i in range(m):
            yone+=y[i]*np.log(h(x[i],theta))
            yzero+=(1-y[i])*np.log(1-h(x[i],theta))
        e=yone+yzero
        e=float(J*(-1/m))
        #print("error",e,"\n")
        
        #update error
        J = e    
        J_log.append(J)
        if iter == max_itr:
            print ('\nMax interactions exceeded!')
    print("\nalpha=",alpha)
    print("max_iterations=",max_itr)
    return theta,J_log

#Train and Parameter tuning
for alpha in [0.2,0.01,0.00075,0.0006,0.0005,0.0002,0.00009,0.000099,0.0000999]:
    print("\n##################################################\n")
    
    #Train the model
    theta,J_log=gradient_descent(alpha,X_train,y_train,20)   
    
    #Plotting cost v/s number of iterations
    plt.title('Cost V/S Iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.plot(range(len(J_log)), J_log)
    plt.show()

    #Train Accuracy
    y_pred=[]
    y_plot=[]
    for i in range(len(X_train)):
        t=(h(X_train[i],theta))
        y_plot.append(t)
        if(t<0.5):
            y_pred.append(0)
        if(t>0.5):
            y_pred.append(1)
    
    #Accuracy analysis
    accuracy = accuracy_score(y_train, y_pred)
    print('Train Accuracy Score:',accuracy*100," %")
    
    #plot training set
    y_pred=[]
    y_plot=[]
    for i in range(len(X_test)):
        t=(h(X_test[i],theta))
        y_plot.append(t)
        if(t<0.5):
            y_pred.append(0)
        if(t>0.5):
            y_pred.append(1)
    
    #Accuracy analysis
    accuracy = accuracy_score(y_test, y_pred)
    print('Test Accuracy Score:',accuracy*100," %")
