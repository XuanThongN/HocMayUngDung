#Chuan bi du lieu
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sympy.logic.inference import KB

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data,
                                                    iris.target,
                                                    test_size=0.3,
                                                    random_state=45)
#Huan luyen mo hinh
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC, NuSVC
from sklearn.neural_network import MLPClassifier
#Khoi tao mo hinh
knn_model = KNeighborsClassifier(n_neighbors=3)
nb_model = GaussianNB()
dt_model = DecisionTreeClassifier()
lsvm_model = LinearSVC()
nu_svm_model = NuSVC()
ann_model = MLPClassifier(hidden_layer_sizes=(30, 40, 30), max_iter=500)
print("Huan luyen cac mo hinh TTNT...")
knn_model.fit(X_train, y_train)
nb_model.fit(X_train, y_train)
dt_model.fit(X_train, y_train)
lsvm_model.fit(X_train, y_train)
nu_svm_model.fit(X_train, y_train)
ann_model.fit(X_train, y_train)
print("KET QUA KIEM THU MO HINH")
y_pred_knn = knn_model.predict(X_test)
y_pred_nb = nb_model.predict(X_test)
y_pred_dt = dt_model.predict(X_test)
y_pred_lsvm = lsvm_model.predict(X_test)
y_pred_nusvm = nu_svm_model.predict(X_test)
y_pred_ann = ann_model.predict(X_test)
from sklearn.metrics import accuracy_score, confusion_matrix
print("MO HINH k-NN")
print("Accuracy: ", accuracy_score(y_true=y_test, y_pred=y_pred_knn))
print("MA TRAN HON HOP k-NN")
print(confusion_matrix(y_true=y_test, y_pred=y_pred_knn))
print("MO HINH NAIVE BAYES")
print("Accuracy: ", accuracy_score(y_true=y_test, y_pred=y_pred_nb))
print("MA TRAN HON HOP NAIVE BAYES")
print(confusion_matrix(y_true=y_test, y_pred=y_pred_nb))
print("MO HINH CAY QUYET DINH")
print("Accuracy: ", accuracy_score(y_true=y_test, y_pred=y_pred_dt))
print("MA TRAN HON HOP CAY QUYET DINH")
print(confusion_matrix(y_true=y_test, y_pred=y_pred_dt))
print("MO HINH Linear SVM")
print("Accuracy: ", accuracy_score(y_true=y_test, y_pred=y_pred_lsvm))
print("MA TRAN HON HOP Linear SVM")
print(confusion_matrix(y_true=y_test, y_pred=y_pred_lsvm))
print("MO HINH SVM PHI TUYEN")
print("Accuracy: ", accuracy_score(y_true=y_test, y_pred=y_pred_nusvm))
print("MA TRAN HON HOP SVM PHI TUYEN")
print(confusion_matrix(y_true=y_test, y_pred=y_pred_nusvm))
print("MO HINH ANN")
print("Accuracy: ", accuracy_score(y_true=y_test, y_pred=y_pred_ann))
print("MA TRAN HON HOP ANN")
print(confusion_matrix(y_true=y_test, y_pred=y_pred_ann))
print("LUU LAI CAC MO HINH DA HUAN LUYEN")
from joblib import dump
dump(knn_model, 'C:/Users/WINDOWS/PycharmProjects/hocmayc2/luutru/knn_model.joblib')
dump(nb_model, 'C:/Users/WINDOWS/PycharmProjects/hocmayc2/luutru/nb_model.joblib')
dump(dt_model, 'C:/Users/WINDOWS/PycharmProjects/hocmayc2/luutru/dt_model.joblib')
dump(lsvm_model, 'C:/Users/WINDOWS/PycharmProjects/hocmayc2/luutru/lsvm_model.joblib')
dump(nu_svm_model, 'C:/Users/WINDOWS/PycharmProjects/hocmayc2/luutru/nu_svm_model.joblib')
dump(ann_model, 'C:/Users/WINDOWS/PycharmProjects/hocmayc2/luutru/ann_model.joblib')
print("KET THUC CHUONG TRINH")
