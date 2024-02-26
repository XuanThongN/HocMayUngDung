import h5py
from sklearn import datasets
digits = datasets.load_digits()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.3, random_state=42)

#Khoi tao mo hinh ANN
from sklearn.neural_network import MLPClassifier
#hidden_layer_sizes la tham so de xay dung
#lop an (100 no-ron trong lop an
#max_iter la tham so de xac dinh
ann_model = MLPClassifier(hidden_layer_sizes=(100,50,150,60,80), max_iter=5000)
#Huan luyen mo hinh tren tap du lieu
ann_model.fit(X_train, y_train)
# Du doan nhan cho tap du lieu test
y_pred = ann_model.predict(X_test)
from sklearn import metrics
#Tinh taon do chinh xac
accuracy = metrics.accuracy_score(y_test, y_pred)
#Tinh toan ma tran nham lan
confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
#In ra do chinh xac va ma tran nham lan
print("Accuracy of the model: ", accuracy)
print("Confusion matrix of the model: ", confusion_matrix)
#Luu mo hinh
from joblib import dump
dump(ann_model, "D:/1. PXU/16. Hoc may va khoa hoc du lieu/3.HocMay01/mo-hinh/mhplann-digits.joblib")

