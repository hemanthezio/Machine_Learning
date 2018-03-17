# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 20:45:40 2018

@author: Hemanth kumar
"""

# Multiple Linear Regression using gradient descent
'''Applying Multiple Linear Regression for house price prediction'''
# Importing the dataset
dataset = pd.read_csv('house.txt')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, [2]].values

# Feature Scaling using standard scalar
def featureScale(X):
    means=np.mean(X,axis=0)
    stdev=np.std(X,axis=0)
    X = (X - means)/stdev
    return X,means,stdev
X,means,stdev=featureScale(X)

#append 1 for theta0
X=np.append(arr=np.ones((len(X),1)).astype(int),values=X,axis=1)
X=X.tolist()
y=y.tolist()

#Gradient Descent Algorithm
def gradient_descent(alpha,x,y,max_itr):
    m =len(x)
    n=len(x[0])
    J_log=list()
    # initial theta
    theta=list()
    for i in range(n):
        theta.append([random.random()])
    
    # total error, J(theta)
    J=np.transpose(np.array(theta))*np.array(x)
    J=np.sum(J,axis=1)
    J=J.reshape(m,1)
    J=J-np.array(y)
    J=np.square(J)
    J=np.sum(J,axis=0)
    J=float(J.reshape(1,1))
    J=J/(2*m)
    J_log.append(J)
    print('initial error=',float(J) )
    
    for i in range(max_itr):
        hx=np.transpose(np.array(theta))*np.array(x)
        hx=np.sum(hx,axis=1)
        hx=hx.reshape(m,1)
        err=hx-np.array(y)
        err=err*np.array(x)
        grad=np.sum(err,axis=0)
        grad=grad.reshape(n,1)
        grad=(alpha/m)*grad
        theta=theta-grad
        
        # mean squared error
        e=np.transpose(np.array(theta))*np.array(x)
        e=np.sum(e,axis=1)
        e=e.reshape(m,1)
        e=e-np.array(y)
        e=e**2
        e=np.sum(e,axis=0)
        e=float(e.reshape(1,1))
        e=e/(2*m)
        
        if abs(J-e) <= 0.0001:
            print ('Converged, iterations: ', i, '!!!')
            return theta,J_log
        
        # update error
        J = e    
        J_log.append(J)
        if iter == max_itr:
            print ('Max interactions exceeded!')
        
    return theta,J_log

#training the data
theta,J_log=gradient_descent(0.01,X,y,10000)

#Plotting cost v/s number of iterations
plt.title('Cost V/S Iterations')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.plot(range(len(J_log)), J_log)
plt.show()

#Performance analysis
y_pred=[]
tmp=np.transpose(np.array(theta))*np.array(X)
tmp=np.sum(tmp,axis=1)
tmp=tmp.reshape(len(X),1)
y_pred=tmp.tolist()

mean=np.mean(y)
y_grid=[]
for i in range(len(y)):
    y_grid.append(mean)
y_grid=np.array(y_grid)
SS_res=float(sum(((np.array(y_pred[i])-np.array(y[i]))**2)for i in range(len(y_pred))))
SS_tot=float(sum(((y_pred[i]-y_grid[i])**2)for i in range(len(y_pred))))

#calculating R SQUARED
R_sq=1-(SS_res/SS_tot)
A_R_sq=1-((1-R_sq)*(len(X))/(len(X)-len(X[0])-1))

print('R Squared =',R_sq)
print('Adjusted R Squared =',A_R_sq)

#prediction
x=[1650,3]
z=np.array(X)
x = (x - means)/stdev
one=np.array([1])
x=np.concatenate((one, x))
x=np.array(x)
x=x.reshape(3,1)

p=theta[0]*x[0]+theta[1]*x[1]+theta[2]*x[2]

print('House price with 1650sq.ft and 3 bedrooms = ',float(p))