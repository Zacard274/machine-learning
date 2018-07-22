import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

path = "ex1.txt"

data = pd.read_csv(path,names=['population','profit'])

data.describe()
print(data.describe())

x=data.population
y=data.profit
plt.figure(figsize=(10,5))
plt.scatter(x,y)
plt.xlabel("Population of City in 10,000s")
plt.ylabel("Profit in $10,000s")
plt.show()

#  等同于：
#data.plot(x='population',y='profit',kind='scatter',figsize=(10,5))
#plt.xlabel("Population of City in 10,000s")
#plt.ylabel("Profit in $10,000s")
#plt.show()