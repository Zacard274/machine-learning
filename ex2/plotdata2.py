import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

path = 'data1.txt'
data = pd.read_csv(path, names = ['exam1', 'exam2', 'admitted'])

print(data.head())
print(data.describe())

positive = data[data.admitted.isin(['1'])]
negetive = data[data.admitted.isin(['0'])]

plt.figure(figsize=(6,5))
plt.scatter(positive['exam1'],positive['exam2'],c='b',marker='o')
plt.scatter(negetive['exam1'],negetive['exam2'],c='r',marker='^')
plt.xlabel('exam1')
plt.ylabel('exam2')
plt.show()



