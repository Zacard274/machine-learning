import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

path = "ex1.txt"

data = pd.read_csv(path,names=['population','profit'])  #索引'population'和'profit'

data.describe()
print(data.describe())

data.insert(0,'ones',1)  # 就是在第一列（0） 添加名字为 ones 的一列数据，他的数值都是 1


#
#     计算代价函数J(θ)
#
def computeCost(x, y, theta):  # 初始化单变量线性回归
    inner = np.power((x * theta.T - y), 2)  # power(x,2)  , 就是将x数组里面的元素都2次方
    return np.sum(inner) / (2 * len(x))

#初始化变量,取最后一列为y，其余为x
cols = data.shape[1]             #shape[]显示行数，列数。 shape[1]即列数
x = data.iloc[:, 0:cols - 1]     # 当索引是字符串那么就用.loc
y = data.iloc[:, cols - 1:cols]  # 当索引是数字那么就用.iloc


# 初始化数据
x = np.matrix(x.values)  # 转化成 矩阵形式
y = np.matrix(y.values)
theta = np.matrix(np.array([0, 0]))  # theta就是一个（1,2）矩阵
costFunction = computeCost(x, y, theta) #计算初始代价函数的值（theat初始值为0）


#
#        批量梯度下降
#
def gradientDescent(x, y, theta, alpha, iters):  # alpha = 学习速率  ，iters = 迭代次数
    temp = np.matrix(np.zeros(theta.shape)) # 初始化一个 θ 临时矩阵(1, 2)
    cost = np.zeros(iters)  # 初始化一个nbarray(多维数组对象），包含每次iters的cost

    for i in range(iters):
        # 利用向量化一步求解
        temp =theta - (alpha / len(x)) * (x * theta.T - y).T * x
        theta = temp
        cost[i] = computeCost(x, y, theta)

    return theta, cost

#   以下是不用Vectorization(向量化)求解梯度下降：

    #for i in range(iters):
    #    error = (x * theta.T) - y
    #    for j in range(parameters):
    #        term = np.multiply(error, x[:, j])  # multiply 对应元素乘法
    #        temp[0, j] = theta[0, j] - ((alpha / len(x)) * np.sum(term))
    #    theta = temp
    #    cost[i] = computeCost(x, y, theta)

    #return theta, cost

#初始化数据
alpha = 0.01
iters = 1000
final_theta,cost = gradientDescent(x,y,theta,alpha,iters)
final_cost = computeCost(x,y,final_theta)
print(f'final_theta={final_theta}')
print(f'final_cost={final_cost}')
print(cost)

x = np.linspace(-5,25,100)  # 横坐标
f = final_theta[0, 0] + (final_theta[0, 1] * x)  # 纵坐标，利润

fig, ax = plt.subplots()
ax.plot(x, f, 'r', label='prediction')
ax.scatter(data.population, data.profit, label='training data')
ax.legend(loc=2)
ax.set_xlabel('population')
ax.set_ylabel('profit')
plt.show()

#
#    由于梯度方程式函数也在每个训练迭代中输出一个代价的向量，所以我们也可以绘制。
#    请注意，线性回归中的代价函数总是降低的 - 这是凸优化问题的一个例子。
#

fig, ax = plt.subplots(figsize=(8,4))
ax.plot(np.arange(iters), cost, 'r')  # np.arange()返回等差数组
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training iters')
plt.show()