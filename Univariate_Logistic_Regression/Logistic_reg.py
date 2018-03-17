# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 20:06:59 2018

@author: Hemanth kumar

"""
#Logistic Regression using gradient descent
'''

Applying Logistic Regression to classifiy customers, whether they will purchase 
the XUV or not based on age and salaray

''' 

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2,3]].values
y = dataset.iloc[:, [4]].values

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

#Hypothesis
def h(x,theta):
    thetax=np.transpose(theta)*x
    thetax=np.sum(thetax,axis=1)
    thetax=float(thetax.reshape(1,1))
    thetax=-thetax
    hx=1/(1+np.exp(thetax))
    return hx

#Gradient descent
def gradient_descent(alpha,x,y,max_itr):
    m =len(x)
    n=len(x[0])
    J_log=list()
    # initial theta
    theta=list()
    for i in range(n):
        theta.append([random.random()])
    theta=np.array(theta)
    
    # total error, J(theta)
    yzero=0
    yone=0
    for i in range(m):
        yone+=y[i]*np.log(h(x[0],theta))
        yzero+=(1-y[i])*np.log(1-h(x[i],theta))
    J=yone+yzero
    J=float(J*(-1/m))
    J_log.append(J)
    print('initial error=',float(J) )
    
    for i in range(max_itr):
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
            yone+=y[i]*np.log(h(x[0],theta))
            yzero+=(1-y[i])*np.log(1-h(x[i],theta))
        e=yone+yzero
        e=float(J*(-1/m))
        
        #update error
        J = e    
        J_log.append(J)
        if iter == max_itr:
            print ('Max interactions exceeded!')
        
    return theta,J_log


#training the data
theta,J_log=gradient_descent(0.01,X_train,y_train,1500)

#Plotting cost v/s number of iterations
plt.title('Cost V/S Iterations')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.plot(range(len(J_log)), J_log)
plt.show()

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

#Visualizing the results
yes=[]
no=[]
for i in range(len(y_test)):
    if(y_test[i]==0):
        no.append([X_test[i,1],X_test[i,2]])
    else:
        yes.append([X_test[i,1],X_test[i,2]])
yes=np.array(yes)
no=np.array(no)
z = np.array([-2,2])
plt.plot(z,(-theta[0]-theta[1]*z)/theta[2],label='Decision Boundry')
plt.scatter(yes[:,0],yes[:,1],color='green',label='Purchased')
plt.scatter(no[:,0],no[:,1],color='red',label='Didnt Purchase')
plt.title('Logistic Regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

#Confusion matrix analysis
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#Accuracy analysis
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy Score:',accuracy)
