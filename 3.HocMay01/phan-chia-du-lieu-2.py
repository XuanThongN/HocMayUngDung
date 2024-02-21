from sklearn import datasets
from sklearn.model_selection import train_test_split

#Tai du lieu
iris = datasets.load_iris()

#Phan chia theo ti le 70 - 30
X_train, X_test, y_train, y_test = train_test_split(iris.data,
                                                    iris.target,
                                                    test_size=0.3,
                                                    random_state=15)
#In kich thuoc
print("Kich thuoc du lieu huan luyen: ", X_train.shape)
print("Kich thuoc du lieu kiem thu: ", X_test.shape)