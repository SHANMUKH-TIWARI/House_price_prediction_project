import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

# Get directories
script_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(script_dir, "..", "Dataset")

# Load CSV files
train = pd.read_csv(os.path.join(dataset_dir, "train.csv"))
test = pd.read_csv(os.path.join(dataset_dir, "test.csv"))
sample_submission = pd.read_csv(os.path.join(dataset_dir, "sample_submission.csv"))

# verify
print("Train shape:", train.shape)
print("Test shape:", test.shape)
print(train.head())

# Check missing values
missing_train = train.isnull().sum().sort_values(ascending=False)
missing_train = missing_train[missing_train > 0]
print("Columns with missing values:\n", missing_train)

numeric_features = train.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = train.select_dtypes(exclude=[np.number]).columns.tolist()

print("Number of numeric features:", len(numeric_features))
print("Number of categorical features:", len(categorical_features))

# SalePrice distribution
sns.histplot(train['SalePrice'], kde=True)
plt.title("SalePrice Distribution")
plt.show()
print("Skewness:", train['SalePrice'].skew())

# Preprocessing
# Log transform target
train["SalePrice_log"] = np.log1p(train["SalePrice"])

# Fill missing categorical values with 'None'
cat_fill_none = ["PoolQC", "MiscFeature", "Alley", "Fence",
                 "GarageType","GarageFinish","GarageQual","GarageCond",
                 "BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinType2",
                 "MasVnrType","FireplaceQu"]

for col in cat_fill_none:
    train[col] = train[col].fillna("None")
    test[col] = test[col].fillna("None")

# Fill missing numerical values with 0
num_fill_zero = ["MasVnrArea","GarageYrBlt"]
for col in num_fill_zero:
    train[col] = train[col].fillna(0)
    test[col] = test[col].fillna(0)

# Fill LotFrontage by neighborhood median
train["LotFrontage"] = train.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
test["LotFrontage"] = test.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))

# Fill Electrical with mode
train["Electrical"] = train["Electrical"].fillna(train["Electrical"].mode()[0])
test["Electrical"] = test["Electrical"].fillna(train["Electrical"].mode()[0])

# One-hot encode categorical features
full_data = pd.concat([train.drop(["SalePrice","SalePrice_log"], axis=1),
                       test], axis=0)
full_data = pd.get_dummies(full_data, drop_first=True)

# Split back to train/test
X_train = full_data.iloc[:train.shape[0], :]
X_test = full_data.iloc[train.shape[0]:, :]
y_train = train["SalePrice_log"]

# Train Random Forest
rf = RandomForestRegressor(
    n_estimators=500,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

# Cross-validation function
def rmse_cv(model, X, y):
    scores = cross_val_score(model, X, y, scoring="neg_root_mean_squared_error", cv=5)
    return -scores

cv_scores = rmse_cv(rf, X_train, y_train)
print("CV RMSE scores:", cv_scores)
print("Mean CV RMSE:", cv_scores.mean())

# Fit on full train
rf.fit(X_train, y_train)

# Predict and save submission
y_pred_log = rf.predict(X_test)
y_pred = np.expm1(y_pred_log)  # back to SalePrice

submission = pd.DataFrame({
    "Id": test["Id"],
    "SalePrice": y_pred
})

# Save CSV
os.makedirs(dataset_dir, exist_ok=True)
submission_path = os.path.join(dataset_dir, "submission.csv")
submission.to_csv(submission_path, index=False)
print(f"Submission saved as {submission_path}")
