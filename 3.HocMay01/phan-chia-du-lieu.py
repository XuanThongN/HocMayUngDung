#from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

#Tai du lieu vao
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]
#Phan chia train - test theo ti le 75% - 25%
X_train, X_test, y_train, y_test = train_test_split(data,
                                                    target,
                                                    test_size=0.25,
                                                    random_state=42)
#In ket qua
print("Kich thuoc du lieu huan luyen: ", X_train.shape)
print("Kich thuoc du lieu kiem thu: ", X_test.shape)