# -*- coding: utf-8 -*-
# @Time    : 2022/4/21 18:56
# @Author  : Chi young Chou
# @FileName:DTRegression2.py
# @Software:PyCharm


import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn import linear_model
plt.rcParams['font.sans-serif'] = ['SimHei']


# Dataset
x = np.array(list(range(1, 11))).reshape(-1, 1)
y = np.array([5.56, 5.70, 5.91, 6.40, 6.80, 7.05, 8.90, 8.70, 9.00, 9.05]).reshape(-1)  # ravel()， 降维
print(x, type(x))
print(y, type(y))

# Fit regression model
X_test = np.arange(0.0, 10.0, 0.01)[:, np.newaxis]  #  升维，和reshape(-1) 、ravel()功能相反
lst = [1, 3, 5]
y1 = []
yr = []
y2 = []
y3 = []
y4 = []
new_list = [y1, yr, y2, y4, y3]

for i in lst:  # 1,3,5
    model = DecisionTreeRegressor(max_depth=i)
    model.fit(x, y)  # 训练深度为1的决策树回归
    new_list[i-1].append(model.predict(X_test))

print('y1的shape为', np.shape(y1))
print('X_test的shape为', np.shape(X_test))


lin_model = linear_model.LinearRegression()
lin_model.fit(x, y)  # 训练线性回归
y4.append(lin_model.predict(X_test))


# Plot the results
# plt.figure(figsize= (20, 8), dpi =80)
plt.figure()
plt.scatter(x, y, s=20, edgecolor="black", c = "darkorange", label="data")
plt.plot(X_test, np.array(y1).T, color="cornflowerblue", label="max_depth=1", linewidth=2)   #  X_test, np.array(y1).T的shpe要一致
plt.plot(X_test, np.array(y2).T, color="yellowgreen", label="max_depth=3", linewidth=2)
plt.plot(X_test, np.array(y3).T, color='red', label='max_depth=5', linewidth=2)
plt.plot(X_test, np.array(y4).T, color='black', label='liner regression', linewidth=2)


plt.xlabel("数据值")
plt.ylabel("目标值")
plt.title("决策树回归")
plt.legend(loc='best')
plt.grid(alpha=0.5)
plt.show()
