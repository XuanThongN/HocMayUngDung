import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LinearRegression

# Doc tap tin csv vao pandas
df = pd.read_csv(os.path.join("D:/1. PXU/16. Hoc may va khoa hoc du lieu/1. Thuc_hanh",
                              'used_car_dataset.csv'), delimiter=',')
# Xuat df ra man hinh
print(df.head())

# Xuat cac ten cot
print(df.columns)

# Lay cac cuo truong du lieu kieu so
df1 = df.select_dtypes(include=['int64', 'float64'])
print(df1.head())

# Xoa cot du lieu dau tien
df1 = df1.drop(df1.columns[0], axis=1)
print(df1.head())

# Xoa cac hang du lieu co chua gia tri NaN
df1 = df1.dropna()

# Chuyen thanh numpy
data = df1.to_numpy()
print(data)
print(data.shape)

# ------------------
X = data[:, :-1]
y = data[:, -1]

model = LinearRegression().fit(X, y)
x = X[3, :].reshape(1, -1)
y = y[3]

y_dudoan = model.predict(x)
print('y = ', y, '; y_dudoan = ', y_dudoan)
