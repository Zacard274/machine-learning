import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

path = "ex1.txt"

data = pd.read_csv(path,names=['population','profit'])  #索引'population'和'profit'

data.insert(0,'ones',1)  # 就是在第一列（0） 添加名字为 ones 的一列数据，他的数值都是 1

cols = data.shape[1]
x = data.iloc[:, 0:cols - 1]     # 当索引是字符串那么就用.loc
y = data.iloc[:, cols - 1:cols]  # 当索引是数字那么就用.iloc

x = np.matrix(x)  # 转化成 矩阵形式
y = np.matrix(y)
theta = np.matrix(np.array([0, 0]))  # theta就是一个（1,2）矩阵

def normalEqn(X,y):
    theta = np.linalg.inv(X.T * X) * X.T * y
    return theta

final_theta = normalEqn(x,y)
print(final_theta)

