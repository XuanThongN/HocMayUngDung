import os
from statistics import LinearRegression
from sklearn.linear_model import LinearRegression
import numpy as np


def readData(folder, filename):
    data = np.loadtxt(os.path.join(folder, filename), delimiter=',')
    print('Original data shape', data.shape)
    X = data[:, 0]
    print('X shape: ', X.shape)
    y = data[:, 1]
    print('y shape: ', y.shape)
    m = y.shape[0]
    print('Number of training examples m = ', m)
    X = np.stack([np.ones(m), X], axis=1)
    print('Modified X shape: ', X.shape)
    return X, y


print('Buoc 1: Doc du lieu X - y')
X, y = readData("D:/1. PXU/16. Hoc may va khoa hoc du lieu/1. Thuc_hanh", "ex1data1.txt")
print('Buoc 2: Huan luyen mo hinh HQTT voi du lieu X, y')
mhhqtt = LinearRegression().fit(X, y)
print('Buoc 3: In ket qua huan luyen mo hinh')
print('Bo thamm so w toi uu: ', mhhqtt.coef_)
print('Buoc 4: Su dung mhhqtt de du doan gia nha')
x = np.array([2000, 6]).reshape(1, -1)
y = mhhqtt.predict(x)
print('Gia tri du doan la:', y)