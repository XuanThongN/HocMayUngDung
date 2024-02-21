import numpy as np
import os


# Dùng thư viện LinearRegression của sklearn
from sklearn.linear_model import LinearRegression

# Đọc file dữ liệu ex1data2.txt ở thư mục D:\1. PXU\16. Hoc may va khoa hoc du lieu\1. Thuc_hanh biết các giá trị cách nhau bởi dấu ,
data =np.loadtxt(os.path.join("D:\1. PXU\16. Hoc may va khoa hoc du lieu\1. Thuc_hanh","ex1data2.txt"),delimiter=",")

# Tách cột dữ liệu đầu tiên đến cột dữ liệu kế cuối của data và lưu vào biến X
X = data[:,:-1]
# Tách cột dữ liệu cuối cùng của data và lưu vào biến y
y = data[:,-1]

# Xây dựng mô hình Linear Regression với bộ dữ liệu X và y
model = LinearRegression().fit(X,y)

# In thông số của mô hình ra màn hình
print("Coefficients: ", model.coef_)
print("Intercept: ", model.intercept_)

# Tạo 1 bộ dữ liệu ngẫu nhiên và lưu vào x để model dự đoán
x = np.array([1, 3]).reshape(1,-1)

# Dự đoán giá trị của x
y_predict = model.predict(x)

# In giá trị dự đoán ra màn hình
print("Predicted y = ", y_predict)

# Lưu lại mô hình model thành file model.pkl
import pickle
pickle.dump(model, open("model.pkl","wb"))


