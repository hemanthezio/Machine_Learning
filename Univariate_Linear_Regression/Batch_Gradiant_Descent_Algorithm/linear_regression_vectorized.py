# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 21:08:54 2018

@author: Hemanth kumar
"""
# -*- coding: utf-8 -*-

# Simple Linear Regression using gradient descent vectorized implementation

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, [1]].values


# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)


def gradient_descent(alpha,x,y,max_itr):
    m =len(x)
    # initial theta
    t0 = random.random()
    t1 = random.random()
    J_log=[]
    # total error, J(theta)
    J=np.sum((t0+x*t1-y)**2)
    J_log.append(J)
    
    print('initial error=',float(J) )
    
    for i in range(max_itr):
        sum1=0
        sum2=0
        sum1=np.sum(t0+t1*x-y)
        sum2=np.sum((t0+t1*x-y)*x)
        
        temp0=t0-((alpha/m)*sum1)
        temp1=t1-((alpha/m)*sum2)
            
        #update theta
        t0=temp0
        t1=temp1
        
        # mean squared error
        e=np.sum((t0+t1*x-y)**2)
        
        if abs(J-e) <= 0.0001:
            print ('Converged, iterations: ', i, '!!!')
            J_log.append(e)
            return t0,t1,J_log
        
        # update error
        J = e
        J_log.append(J)
    
        if iter == max_itr:
            print ('Max interactions exceeded!')
        
    return t0,t1,J_log

#training the data
theta0,theta1,J_log =gradient_descent(0.01,X_train,y_train,10000)

plt.plot(J_log,range(len(J_log)))
plt.title("Cost VS Iterations")
plt.xlabel("iterations")
plt.ylabel("cost J")
plt.legend()
plt.show()

#plot training set
y_pred=[]
y_pred=theta0+theta1*X_train

plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, y_pred, color = 'blue')
plt.title('Salary vs Experience (Training set) using gradient descent ')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#predict the test set values
y_pred=[]
y_pred=theta0+theta1*X_test

plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_test, y_pred, color = 'blue')
plt.title('Salary vs Experience (Test set) using gradient descent ')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#Evaluating performance
mid=int(np.mean(y_test))

#plotting average regression line
y_grid=[]
for i in range(len(y_test)):
    y_grid.append(mid)

plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_test, y_grid , color = 'blue')
plt.title('Salary vs Experience (Test set) plotting average regression line to calculate R^2 ')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

SS_res=float((np.sum((y_pred-y_test)**2)))
y_tot=mid
SS_tot=float((np.sum((y_tot-y_test)**2)))

#calculating R SQUARED
R_sq=1-(SS_res/SS_tot)

print('R SQUARED = ',R_sq)
