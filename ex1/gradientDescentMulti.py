import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

path = "ex2.txt"

data = pd.read_csv(path,names=['size','bedrooms','price'])


#   特征归一化。 这个对于pandas来说很简单
data = (data - data.mean()) / data.std()
data.insert(0,'ones',1)
print(data.head())

cols = data.shape[1]
m = data.shape[0]
x = data.iloc[:,0:cols-1]
y = data.iloc[:,cols-1:cols]


def computeCost(x,y,theta):
    inner = np.power((x * theta.T - y), 2)
    return np.sum(inner) / (2 * m)


x = np.matrix(x)
y = np.matrix(y)
print(x.shape[1])
theta = np.matrix(np.zeros(x.shape[1]))  # theta就是一个（1,cols-1）矩阵
print(theta)


def gradientDescentMulti(x, y, theta, alpha, iters):
    temp = np.zeros(theta.shape[1])
    cost = np.zeros(iters)    #cost.shape = (iters)而不是（1，iters),是向量

    for i in range(iters):
        temp = theta - (alpha / m) * (x * theta.T - y).T * x
        theta = temp

        cost[i] = computeCost(x, y, theta)  #如果cost.shape=(1,iters),此处应为cost[0,i]


    return theta,cost

alpha = 0.01
iters = 1000
final_theta , cost = gradientDescentMulti(x,y,theta,alpha,iters)
final_cost = computeCost(x,y,final_theta)
print(f'final_theta={final_theta}') 
print(f'final_cost={final_cost}')
print(len(cost))


fig, ax = plt.subplots(figsize=(8,5))
ax.plot(np.arange(iters), cost,'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')
plt.show()
