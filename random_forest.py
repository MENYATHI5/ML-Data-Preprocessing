# Level 3 Task 1: Random Forest Classifier

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pandas as pd

# Load dataset
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

# Create Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Cross Validation
cv_scores = cross_val_score(
    model,
    X,
    y,
    cv=5
)

print("✅ Random Forest Classifier Complete")

print("\nCross Validation Scores:")
print(cv_scores)

print("\nAverage CV Accuracy:")
print(cv_scores.mean())

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature Importance
importance = pd.DataFrame({
    "Feature": iris.feature_names,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance:")
print(importance)