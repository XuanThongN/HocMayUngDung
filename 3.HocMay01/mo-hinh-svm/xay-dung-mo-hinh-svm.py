from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from joblib import dump

def train_svm_model():
    # Load the digits dataset
    digits = datasets.load_digits()

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.3, random_state=42)

    # Initialize the SVM model with linear kernel
    svm_model = svm.LinearSVC()

    # Train the model on the training set
    svm_model.fit(X_train, y_train)

    # Predict the labels for the test set
    y_pred = svm_model.predict(X_test)

    # Calculate the accuracy
    accuracy = svm_model.score(X_test, y_test)

    # Calculate the confusion matrix
    confusion_matrix = metrics.confusion_matrix(y_test, y_pred)

    # Print the accuracy and confusion matrix
    print("Accuracy of the model: ", accuracy)
    print("Confusion matrix of the model: ", confusion_matrix)

    # Save the model
    dump(svm_model, "D:/1. PXU/16. Hoc may va khoa hoc du lieu/3.HocMay01/mo-hinh/mhplsvm-digits.joblib")

# Call the function to train the SVM model
train_svm_model()