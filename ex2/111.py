import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


a = np.array([[1, 1, 1], [1, 1, 1],[0, 0, 0]])
b = np.array([[0, 0, 0], [0, 0, 0],[1, 1, 1]])


print(np.sum(a+b))
print(a+b)
c = a+b
print(np.mean(c))

print('---'*20)


m=np.array([1,2,3,4])
n=np.array([[1],[2],[3],[4]])
print(m.shape)
print(n.shape)
print(m@n)
print(m[1])