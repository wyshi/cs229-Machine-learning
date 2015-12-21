import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

q1x = np.loadtxt('q1x.dat')
q1y = np.loadtxt('q1y.dat')


start = np.array([0,0,0])
q1x = np.insert(q1x, 0, 1, axis=1)

def g(theta,x):
    a=1+np.exp(-np.dot(theta,x.T))
    return float(1)/a

def grad(theta,x,y):
    total = np.zeros(3)
    
    for i in range(len(y)):
        term = (y[i]-g(theta,x[i,:]))*x[i,:]
        total+=term

    return total


def H(theta,x):
    total = np.zeros((3,3))
    for i in range(x.shape[0]):
        term = -g(theta,x[i,:])*(1-g(theta,x[i,:]))*np.mat(x[i,:]).T*np.mat(x[i,:])
        total+=term
    return total


max_iter = 50
start = np.array([0,0,0])


times = 0
for i in range(max_iter):

    new = np.array(np.mat(start).T - (np.mat(inv(H(start,q1x)))*np.mat(grad(start,q1x,q1y)).T))
    new=new.reshape(3,)

    start =new
    times+=1
    if(grad(start,q1x,q1y).all()==0):
        break

print "how many iterations?"
print times


x1 = q1x[:,1]
x2 = q1x[:,2]
y=q1y

plt.scatter(x1,x2,c=y)
slope,intercept=-0.6489,2.2361
plt.plot(x1,x1*slope+intercept,'r')

plt.savefig('hm1_q1')

#plt.show()






