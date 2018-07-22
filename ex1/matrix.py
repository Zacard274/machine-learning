import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os, sys


a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12],[13,14,15,16]])
print(f'a[:2]={a[:2]}')       #前两行
print(f'a[2:]={a[2:]}')       #后几行
print(f'a[1,:]={a[1,:]}')     #第2行
print(f'a[3]={a[3]}')         #同上，简写
print(f'a[:,1]={a[:,1]}')     #第2列

print(f'a[:2,1:3]={a[:2,1:3]}')
print(f'a[1,2]={a[1,2]}')

print('=='*20)

print(a[:,1])      #不是矩阵，一维
print(a[:,1:2])    #4*1矩阵，二维

print('--'*20)

c=a[1,:]           #一维
d=a[1:2,:]         #二维
print(c,c.shape)
print(d,d.shape)
print(type(c))
print(type(d))
print(c[2])
#print(c[0,2])  报错
print(d[0,2])
print(d[:,2])


print('=='*20)

h = np.zeros(5)
j = np.matrix(np.zeros(5))
print(h)
print(j)

print(h[1])

print(j[0,1])

print(np.matrix([[1,2]]))
print(np.array(np.matrix([[1,2]])))


print('++'*20)

m = np.array([1,2,3,4])
n = np.array([[1],[2],[3],[4]])

print(m.shape)
print(n.shape)
print(m*n)
print(n*m)
print(type(m),type(n))

print('^^'*60)

y = np.array([1,2,3,4])
z = np.array([[1,2,3,4]])
print(y.shape)
print(z)
print(z.shape)
x = z[0]
print(x.shape)



def gradientDescent(x,y,alpha,theta,iters):
    temp = np.zeros(x.shape[1])
    cost = np.matrix(np.zeros(iters))

    for i in range(iters):
        temp = theta - alpha * ((sigmoid(x @ theta) - y).T @ x).T / m
        theta = temp
        cost[0,i] = costfunction(x,y,theta)

    return theta,cost

alpha = 0.0003
iters = 500
final,cost = gradientDescent(x,y,alpha,theta,iters)
print(f'final={final}')
#print(cost)