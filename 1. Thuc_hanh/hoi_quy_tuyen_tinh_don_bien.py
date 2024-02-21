import os
import numpy as np
from matplotlib import pyplot as plt

def readData(folder, filename):
    data = np.loadtxt(os.path.join(folder, filename), delimiter=',')
    print('Original data shape', data.shape)
    X = data[:,0]
    print('X shape: ', X.shape)
    y = data[:,1]
    print('y shape: ', y.shape)
    m = y.shape[0]
    print('Number of training examples m = ', m)
    X = np.stack([np.ones(m), X], axis=1)
    print('Modified X shape: ', X.shape)
    return X, y

def computeLoss(X, y, w):
    m = y.shape[0]
    J = 0
    h = np.dot(X, w)
    J = (1/(2*m))*np.sum(np.square(h - y))
    return J

def gradientDescent(X, y, w, alpha, n):
    m = y.shape[0]
    J_history = []
    w_optimal = w.copy()
    print('w_optimal shape: ', w_optimal.shape)
    for i in range(n):
        w_optimal = w_optimal - (alpha/m)*(np.dot(X, w_optimal) - y).dot(X)
        J_history.append(computeLoss(X, y, w_optimal))
    return w_optimal, J_history

def visualizeDataAndModel(X, y, w_optimal):
    fig = plt.figure()
    plt.plot(X[:,1], y, 'g^')
    plt.plot(X[:, 1], np.dot(X, w_optimal), 'r-')
    plt.legend(['Raw Data', 'Linear regression'])
    plt.ylabel('Profit in $10,000')
    plt.xlabel('Population of City in 10,000s')
    plt.show()

def main():
    w = np.zeros(2)
    n = 1500
    alpha = 0.01
    X, y = readData("D:/1. PXU/16. Hoc may va khoa hoc du lieu/1. Thuc_hanh", "ex1data1.txt")
    w, J_history = gradientDescent(X, y, w, alpha, n)
    print("Optimal weights are: ", w)
    print("Loss function: ", J_history[-1])
    visualizeDataAndModel(X, y, w)

if __name__ == '__main__':
    main()