from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

#Lấy dữ liệu train - test
digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data,
                                                    digits.target,
                                                    test_size=0.3,
                                                    random_state=99)
#Xây dựng mô hình k-NN với k = 3
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

#Đánh gía mo hinh
y_pred = dt.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))

#Lưu mô hình
from joblib import dump
dump(dt, "D:/1. PXU/16. Hoc may va khoa hoc du lieu/3.HocMay01/mo-hinh/mhdt-digits.joblib")
print("Luu mo hinh thanh cong")