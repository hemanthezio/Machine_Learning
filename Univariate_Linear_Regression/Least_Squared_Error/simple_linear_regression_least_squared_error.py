# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 20:53:21 2017

@author: Hemanth kumar
"""
''' Using Simple Linear Regression (Least squared error method) to predict the salary based on years of experience '''

# Simple Linear Regression using Least squared error method

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state = 0)

# calculate b0,b1
sx=0
sy=0
sxy=0
sxx=0
for i in range(len(x_train)):
    sx+=x_train[i]
    sy+=y_train[i]
    sxy=sxy+(x_train[i]*y_train[i])
    sxx=sxx+(x_train[i]*x_train[i])

n=len(x_train)
ssxy=sxy-((sx*sy)/n)
ssxx=sxx-((sx*sx)/n)
b1=ssxy/ssxx
b0=(sy-(b1*sx))/n

# plotting training set
y_pred=[]
for i in range(len(x_train)):
    y_pred.append(int(b0+b1*x_train[i]))
Y=np.array(y_pred)
plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, y_pred, color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#plotting test set
y_pred=[]
for i in range(len(x_test)):
    y_pred.append(int(b0+b1*x_test[i]))
Y=np.array(y_pred)
plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_test, y_pred, color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

