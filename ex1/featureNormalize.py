import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

path = "ex1.txt"

data = pd.read_csv(path,names=['population','profit'])

data.insert(0,'ones',1)  # 就是在第一列（0） 添加名字为 ones 的一列数据，他的数值都是 1

#def computeCost(x, y, theta):  # 初始化单变量线性回归
#    inner = np.power(((x * theta.T) - y), 2)  # power(x,2)  , 就是将x数组里面的元素都2次方
#    return np.sum(inner) / (2 * len(x))

#初始化变量,取最后一列为y，其余为x
cols = data.shape[1]             #shape[]显示行数，列数。 shape[1]即列数
x = data.iloc[:, 0:cols - 1]     # x是所有的行,去掉最后一列
y = data.iloc[:, cols - 1:cols]  # y就是最后一列数据

# 初始化数据
x = np.matrix(x.values)  # 转化成 矩阵形式
y = np.matrix(y.values)
mu = np.matrix(np.zeros(x.shape[1]))
sigma = np.matrix(np.zeros(x.shape[1]))

mu = np.mean(x, axis=0)  # 每一列的均值
sigma = np.std(x, axis=0)  # 每一列的标准差

for i in range(x.shape[1]):
    x[:,i] = (x[:,i] - mu[0,i]) / sigma[0,i]

print(x)


#
#   全篇简写为：
#

#path = 'ex1.txt'
#data = pd.read_csv(path, names=['population','profit'])
#data = (data - data.mean()) / data.std()


