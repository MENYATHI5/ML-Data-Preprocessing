import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix
)

# Load datasets
train_df = pd.read_csv("churn-bigml-80.csv")
test_df = pd.read_csv("churn-bigml-20.csv")

# Combine datasets
df = pd.concat([train_df, test_df], ignore_index=True)

# Encode categorical columns
le = LabelEncoder()

for col in df.select_dtypes(include="object").columns:
    df[col] = le.fit_transform(df[col])

# Features and target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Use provided train/test split
X_train = X.iloc[:len(train_df)]
X_test = X.iloc[len(train_df):]

y_train = y.iloc[:len(train_df)]
y_test = y.iloc[len(train_df):]

# Train model
model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("✅ Logistic Regression Complete")

print("\nAccuracy:",
      accuracy_score(y_test, y_pred))

print("\nPrecision:",
      precision_score(y_test, y_pred))

print("\nRecall:",
      recall_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))