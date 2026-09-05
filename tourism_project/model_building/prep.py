
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = "tourism_project/data/tourism.csv"

# Load dataset
df = pd.read_csv(DATA_PATH)

# Remove unnecessary columns
df = df.drop(columns=["Unnamed: 0", "CustomerID"], errors="ignore")

# Separate features and target
X = df.drop(columns=["ProdTaken"])
y = df["ProdTaken"]

# Split into training and testing sets
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Save splits in the project root
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Data preparation completed successfully!")
print("Xtrain shape:", Xtrain.shape)
print("Xtest shape:", Xtest.shape)
print("ytrain shape:", ytrain.shape)
print("ytest shape:", ytest.shape)
print("\nTraining target distribution:")
print(ytrain.value_counts())
print("\nTesting target distribution:")
print(ytest.value_counts())
