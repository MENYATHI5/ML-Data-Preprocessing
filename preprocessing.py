#  Data Preprocessing for Machine Learning

# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset (space-separated, no header)
df = pd.read_csv("house Prediction Data Set.csv", sep="\s+", header=None)

# Assign column names
df.columns = [
    "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE",
    "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"
]

# Display dataset info
print("First 5 rows:\n", df.head())

# Check missing values
print("\nMissing Values:\n", df.isnull().sum())

# Handle missing values (fill with mean)
df = df.fillna(df.mean())

# Split features and target
X = df.drop("MEDV", axis=1)
y = df["MEDV"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Final output
print("\n Preprocessing Complete!")
print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)

print("\n Data preprocessing ready for machine learning models.")