from sklearn.linear_model import Lasso
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from joblib import dump, load

#Tai du lieu vao
print("Tai du liẹu vao")
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

#Phan chia du lieu train - test theo ti le 75% - 25%
X_train, X_test, y_train, y_test = train_test_split(data,
                                                    target,
                                                    test_size=0.25,
                                                    random_state=42)

print("Huấn luyện mô hình")
model = Lasso()
model.fit(X_train, y_train)
print("Hệ số chặn w0: ", model.intercept_)
print("Hệ số dốc wi: ", model.coef_)

print("Kiem thu mô hình")
y_pred = model.predict(X_test)
print("MAE = ", mean_absolute_error(y_test, y_pred))
print("MSE = ", mean_squared_error(y_test, y_pred))

print("Lưu mô hình vào file")
dump(model, "D:/1. PXU/16. Hoc may va khoa hoc du lieu/3.HocMay01/mo-hinh/mhhqlasso.joblib")

print("Tai mô hình từ file")
model2 = load("D:/1. PXU/16. Hoc may va khoa hoc du lieu/3.HocMay01/mo-hinh/mhhqlasso.joblib")
print("Success")