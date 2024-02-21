import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LinearRegression

def readDataFromCSV(folder, filename):
    # Doc tap tin csv vao pandas
    df = pd.read_csv(os.path.join(folder, filename), delimiter=',')
    # Lay cac cuo truong du lieu kieu so
    df1 = df.select_dtypes(include=['int64', 'float64'])
    # Xoa cot du lieu dau tien
    df1 = df1.drop(df1.columns[0], axis=1)
    # Xoa cac hang du lieu co chua gia tri NaN
    df1 = df1.dropna()
    # Chuyen thanh numpy
    data = df1.to_numpy()
    return data

data = readDataFromCSV("D:/dataset/salary", 'Salary_dataset.csv')
X = data[:, :-1]
y = data[:,-1]

model = LinearRegression().fit(X, y)
x = X[3, :].reshape(1, -1)
y = y[3]

y_dudoan = model.predict(x)
print('y = ', y, '; y_dudoan = ', y_dudoan)