from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_lfw_people

lfw = fetch_lfw_people()

X_train, X_test, y_train, y_test = train_test_split(lfw.data,
                                                    lfw.target,
                                                    test_size=0.3,
                                                    random_state=55)
#In kich thuoc
print("Kich thuoc du lieu huan luyen: ", X_train.shape)
print("Kich thuoc du lieu kiem thu: ", X_test.shape)