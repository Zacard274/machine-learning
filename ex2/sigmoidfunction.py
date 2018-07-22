import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

path = "data1.txt"

data = pd.read_csv(path,names=['exam1', 'exam2', 'admitted'])
data.insert(0,'ones',1)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def costfunction(x,y,theta):
    #return -np.mean(y * np.log(sigmoid(x @ theta)) + (1 - y) * np.log(1 - sigmoid(x @ theta)))
    return -np.sum(y * np.log(sigmoid(x @ theta)) + (1 - y) * np.log(1 - sigmoid(x @ theta))) / m
    # 在numpy中，@等同于 .matmul()函数，即矩阵相乘。
    # *代表的是并不是矩阵的乘法规则，而是简单的数量积，即对应位置元素相乘后的积相加。


n = data.shape[1]
m = data.shape[0]
x = data.iloc[:,0:n-1].as_matrix()  #也可以写成X=data.iloc[:,:-1]
y = data.iloc[:,n-1].as_matrix()  #as.matrix()把其转换成numpy.array
theta = np.zeros(x.shape[1])

print(costfunction(x,y,theta))
print(x.shape, theta.shape, y.shape)


def gradient(theta, X, y):
    return (X.T @ (sigmoid(X @ theta) - y))/len(X)

f_gradient = gradient(theta,x,y)
print(f'f_gradient={f_gradient}')

#这里使用fimin_tnc或者minimize方法来拟合，minimize中method可以选择不同的算法来计算，其中包括TNC

import scipy.optimize as opt

result = opt.fmin_tnc(func=costfunction(x,y,theta), x0=theta, fprime=gradient, args=(x, y))
print(result)