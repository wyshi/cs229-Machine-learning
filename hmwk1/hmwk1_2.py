import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

x = np.loadtxt('q2x.dat')
y = np.loadtxt('q2y.dat')
x=x.reshape(100,1)
x = np.insert(x, 0, 1, axis=1)
theta = np.dot(inv(np.dot(x.T,x)),np.dot(x.T,y))
print x[1,:]
print theta

tao = 1000

def computeW(x0,x):
    w=[np.exp(-np.dot((x0-x[i,:]).T,(x0-x[i,:]))**2/(2*tao*tao)) for i in range(x.shape[0])]
    w=np.array(w)
    w=np.diag(w)
    return w

def fit(x):
    hx = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        w=computeW(x[i,:],x)
        theta2 = inv(np.mat(x.T)*np.mat(w)*np.mat(x))*(np.mat(x.T)*np.mat(w)*np.mat(y).T)
        hx[i] = np.dot(theta2.T,x[i,:])[0,0]
    return hx

hx = fit(x)


x1 = x[:,1]

plt.scatter(x1,y)
slope,intercept=theta[1],theta[0]
plt.plot(x1,x1*slope+intercept,'r')

plt.scatter(x1,hx,c='r')

plt.savefig('hm1_q2_tao1000')
print "the bigger tao, the closer to the straignt line, which is when w(i)=1"
plt.show()


















