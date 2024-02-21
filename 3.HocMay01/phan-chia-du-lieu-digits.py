from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

digits = load_digits()
a
X_train, X_test, y_train, y_test = train_test_split(digits.data,
                                                    digits.target,
                                                    test_size=0.3,
                                                    random_state=45)
#In kich thuoc
print("Kich thuoc du lieu huan luyen: ", X_train.shape)
print("Kich thuoc du lieu kiem thu: ", X_test.shape)