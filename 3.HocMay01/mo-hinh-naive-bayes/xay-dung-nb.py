from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

#Lấy dữ liệu train - test
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data,
                                                    iris.target,
                                                    test_size=0.3,
                                                    random_state=42)
#Xây dựng mô hình k-NN với k = 3
nb = GaussianNB()
nb.fit(X_train, y_train)

#Đánh gía mo hinh
y_pred = nb.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))

#Lưu mô hình
from joblib import dump
dump(nb, "D:/1. PXU/16. Hoc may va khoa hoc du lieu/3.HocMay01/mo-hinh/mhnb-3.joblib")
print("Luu mo hinh thanh cong")