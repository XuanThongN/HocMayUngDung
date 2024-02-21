from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine

wine = load_wine()

X_train, X_test, y_train, y_test = train_test_split(wine.data,
                                                    wine.target,
                                                    test_size=0.3,
                                                    random_state=27)
#In kich thuoc
print("Kich thuoc du lieu huan luyen: ", X_train.shape)
print("Kich thuoc du lieu kiem thu: ", X_test.shape)