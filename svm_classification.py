# Level 3 Task 2: Support Vector Machine (SVM)

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# Load Iris dataset
iris = load_iris()

X = iris.data
y = iris.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Linear Kernel
linear_svm = SVC(kernel="linear")

linear_svm.fit(X_train, y_train)

linear_pred = linear_svm.predict(X_test)

linear_accuracy = accuracy_score(y_test, linear_pred)

# RBF Kernel
rbf_svm = SVC(kernel="rbf")

rbf_svm.fit(X_train, y_train)

rbf_pred = rbf_svm.predict(X_test)

rbf_accuracy = accuracy_score(y_test, rbf_pred)

print("✅ SVM Classification Complete")

print("\nLinear Kernel Accuracy:")
print(f"{linear_accuracy:.2f}")

print("\nRBF Kernel Accuracy:")
print(f"{rbf_accuracy:.2f}")

print("\nClassification Report (RBF Kernel):")
print(classification_report(y_test, rbf_pred))