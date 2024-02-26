from sklearn.datasets import load_wine
#Doc va phan chia du lieu
wine = load_wine()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(wine.data,
                                                    wine.target,
                                                    test_size=0.3,
                                                    random_state=13)
#Huan luyen mo hinh voi tap du lien tain (X_train, y_train)
from sklearn.svm import LinearSVC
lsvc_model = LinearSVC()
lsvc_model.fit(X_train, y_train)
from sklearn.svm import SVC
svc_model = SVC()
svc_model.fit(X_train, y_train)
from sklearn.svm import NuSVC
nusvc_model = NuSVC(gamma='auto')
nusvc_model.fit(X_train, y_train)
#Danh gia hieu nang cua mo hinh
from sklearn.metrics import accuracy_score, confusion_matrix
y_pred1 = lsvc_model.predict(X_test)
y_pred2 = svc_model.predict(X_test)
y_pred3 = nusvc_model.predict((X_test))
print("KET QUA DU DOAN CUA MO HINH LINEAR SVC")
print("Accuracy score: ", accuracy_score(y_pred1, y_test))
print("Confusion matrix", confusion_matrix(y_test, y_pred1))
print("KET QUA DU DOAN CUA MO HINH SVC")
print("Accuracy score: ", accuracy_score(y_test, y_pred2))
print("Confusion matrix", confusion_matrix(y_test, y_pred2))
print("KET QUA DU DOAN CUA MO HINH NUSVC")
print("Accuracy score: ", accuracy_score(y_test, y_pred3))
print("Confusion matrix", confusion_matrix(y_test, y_pred3))
#Luu mo hinh lai
from joblib import dump
dump(lsvc_model,
     "D:/1. PXU/16. Hoc may va khoa hoc du lieu/3.HocMay01/mo-hinh/lsvc_wine.joblib")
dump(svc_model,
     "D:/1. PXU/16. Hoc may va khoa hoc du lieu/3.HocMay01/mo-hinh/svc_wine.joblib")
dump(nusvc_model,
     "D:/1. PXU/16. Hoc may va khoa hoc du lieu/3.HocMay01/mo-hinh/nusvc_wine.joblib")
print("Ket thuc chuong trinh")