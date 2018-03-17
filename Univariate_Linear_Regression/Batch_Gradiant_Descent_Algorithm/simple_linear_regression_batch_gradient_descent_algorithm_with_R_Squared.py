# -*- coding: utf-8 -*-
"""
Created on Sun Dec 31 11:47:58 2017

@author: Hemanth kumar
"""

# Simple Linear Regression using gradient descent with performance evaluation

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


def gradient_descent(alpha,x,y,max_itr):
    m =len(x)
    
    # initial theta
    t0 = random.random()
    t1 = random.random()
    
    # total error, J(theta)
    J = sum([(t0 + t1*x[i] - y[i])**2 for i in range(m)])
    
    print('initial error=',float(J) )
    
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
    

plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_test, y_pred, color = 'blue')
plt.title('Salary vs Experience (Test set) using gradient descent ')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#Evaluating performance
maxx=float(max(y_test))
minn=float(min(y_test))
mid=(maxx+minn)/2


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

SS_res=float(sum(((y_pred[i]-y_test[i])**2)for i in range(len(y_pred))))
#y=mx+c m=1 theta=90 c=mid
#y=x+mid

y_tot=X_test[:]
y_tot=y_tot+mid

SS_tot=float(sum(((y_tot[i]-y_test[i])**2)for i in range(len(y_pred))))

#calculating R SQUARED
R_sq=1-(SS_res/SS_tot)

print('R SQUARED = ',R_sq)
