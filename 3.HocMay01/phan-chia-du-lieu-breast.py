from sklearn.model_selection import train_test_split
from sklearn.datasets  import load_breast_cancer

breast = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(breast.data,
                                                    breast.target,
                                                    test_size=0.3,
                                                    random_state=15)
#In kich thuoc
print("Kich thuoc du lieu huan luyen: ", X_train.shape)
print("Kich thuoc du lieu kiem thu: ", X_test.shape)