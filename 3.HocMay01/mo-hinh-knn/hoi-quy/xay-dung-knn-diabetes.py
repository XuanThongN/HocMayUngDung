#Buoc 1: Khai bao cac thu vien can thiet
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

#Buoc 2: Tai va phan chia train - test
diabetes = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(diabetes.data,
                                                    diabetes.target,
                                                    test_size=0.3,
                                                    random_state=42)
#Buoc 3: Xay dung mo hinh voi tap du lieu huan luyen train
knn_model = KNeighborsRegressor(n_neighbors=3)
knn_model.fit(X_train, y_train)

#Buoc 4: Danh gia hieu nang cua mo hinh voi tap du lieu kiem thu test
y_pred = knn_model.predict(X_test)
print("Chi so MSE cua mo hinh la: ", mean_squared_error(y_test, y_pred))
print("Chi so MAE cua mo hinh la: ", mean_absolute_error(y_test, y_pred))

#Buoc 5: Luu lai mo hinh
from joblib import dump
dump(knn_model, "D:/1. PXU/16. Hoc may va khoa hoc du lieu/3.HocMay01/mo-hinh/mhplknn-diabetes-3.joblib")
print("Luu mo hinh thanh cong")

