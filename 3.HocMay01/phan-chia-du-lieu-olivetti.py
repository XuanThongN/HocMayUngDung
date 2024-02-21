from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_olivetti_faces

olivetti = fetch_olivetti_faces()

X_train, X_test, y_train, y_test = train_test_split(olivetti.data,
                                                    olivetti.target,
                                                    test_size=0.3,
                                                    random_state=25)
#In kich thuoc
print("Kich thuoc du lieu huan luyen: ", X_train.shape)
print("Kich thuoc du lieu kiem thu: ", X_test.shape)