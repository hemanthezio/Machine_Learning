# -*- coding: utf-8 -*-
"""
Created on Sun Dec 31 11:47:58 2017

@author: Hemanth kumar
"""
''' Using Simple Linear Regression (Gradient Descent algorithm) to predict the salary based on years of experience '''

# Simple Linear Regression using gradient descent

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values


# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

#Gradient Descent Algorithm
def gradient_descent(alpha,x,y,max_itr):
    m =len(x)
    
    # initial theta
    t0 = random.random()
    t1 = random.random()
    
    # total error, J(theta)
    J = sum([(t0 + t1*x[i] - y[i])**2 for i in range(m)])
    
    print('initial error=',J )
    
    for i in range(max_itr):
        sum1=0
        sum2=0
            
        sum1=sum((t0+t1*x[j]-y[j]) for j in range (m))  
        sum2=sum(((t0+t1*x[j]-y[j])*x[j]) for j in range(m))
            
        temp0=t0-((alpha/m)*sum1)
        temp1=t1-((alpha/m)*sum2)
            
        #update theta
        t0=temp0
        t1=temp1
        
        # mean squared error
        e = sum( [ (t0 + t1*x[i] - y[i])**2 for i in range(m)] )
        
        if abs(J-e) <= 0.0001:
            print ('Converged, iterations: ', i, '!!!')
            return t0,t1
        
        # update error
        J = e    
    
        if iter == max_itr:
            print ('Max interactions exceeded!')
        
    return t0,t1

#training the data
theta0,theta1=gradient_descent(0.01,X_train,y_train,10000)

#plot training set
y_pred=[]
for i in range(len(X_train)):
    y_pred.append(theta0+theta1*X_train[i])

plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, y_pred, color = 'blue')
plt.title('Salary vs Experience (Training set) using gradient descent ')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#predict the test set values
y_pred=[]
for i in range(len(X_test)):
    y_pred.append(theta0+theta1*X_test[i])

#plotting test set
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_test, y_pred, color = 'blue')
plt.title('Salary vs Experience (Test set) using gradient descent ')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


