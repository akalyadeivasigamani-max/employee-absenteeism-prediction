import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================================================
# 1. LOAD DATASET
# =========================================================

df = pd.read_csv("Employee.csv")

print("Original Dataset Shape:", df.shape)

print("\nColumn Names:")
print(df.columns.tolist())


# =========================================================
# 2. DATA CLEANING
# =========================================================

# Remove duplicate rows
df = df.drop_duplicates()

# Remove completely empty rows
df = df.dropna(how="all")

print("\nAfter Cleaning:", df.shape)


# =========================================================
# 3. REMOVE UNNECESSARY COLUMNS
# =========================================================

columns_to_drop = [
    "ID",
    "Employee ID",
    "Employee Number"
]

for col in columns_to_drop:
    if col in df.columns:
        df = df.drop(columns=[col])


# EmployeeID is also an identifier, not a useful
# prediction feature
if "EmployeeID" in df.columns:
    df = df.drop(columns=["EmployeeID"])


# =========================================================
# 4. DEFINE TARGET COLUMN
# =========================================================

target_column = "AbsenteeismDays"

if target_column not in df.columns:

    print("\nAvailable columns:")
    print(df.columns.tolist())

    raise ValueError(
        f"Target column '{target_column}' not found in dataset."
    )


# =========================================================
# 5. CONVERT TARGET TO NUMERIC
# =========================================================

df[target_column] = pd.to_numeric(
    df[target_column],
    errors="coerce"
)

# Remove rows where target is missing
df = df.dropna(subset=[target_column])


# =========================================================
# 6. SEPARATE FEATURES AND TARGET
# =========================================================

X = df.drop(columns=[target_column])

y = df[target_column]


# =========================================================
# 7. HANDLE MISSING VALUES
# =========================================================

# Numerical columns
numeric_columns = X.select_dtypes(
    include=["int64", "float64", "int32", "float32"]
).columns

for col in numeric_columns:

    X[col] = X[col].fillna(
        X[col].median()
    )


# Categorical columns
categorical_columns = X.select_dtypes(
    include=["object", "category"]
).columns

for col in categorical_columns:

    if X[col].isnull().any():

        X[col] = X[col].fillna(
            X[col].mode()[0]
        )


# =========================================================
# 8. ENCODE CATEGORICAL FEATURES
# =========================================================

encoders = {}

for col in categorical_columns:

    encoder = LabelEncoder()

    X[col] = encoder.fit_transform(
        X[col].astype(str)
    )

    encoders[col] = encoder


# =========================================================
# 9. TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Data:", X_train.shape)
print("Testing Data :", X_test.shape)


# =========================================================
# 10. RANDOM FOREST REGRESSION MODEL
# =========================================================

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)


# =========================================================
# 11. TRAIN MODEL
# =========================================================

model.fit(
    X_train,
    y_train
)

print(
    "\nRandom Forest Regression Model "
    "Trained Successfully!"
)


# =========================================================
# 12. PREDICTION
# =========================================================

y_pred = model.predict(X_test)


# =========================================================
# 13. MODEL EVALUATION
# =========================================================

mae = mean_absolute_error(
    y_test,
    y_pred
)

mse = mean_squared_error(
    y_test,
    y_pred
)

rmse = np.sqrt(mse)

r2 = r2_score(
    y_test,
    y_pred
)


print("\n==============================")
print("MODEL EVALUATION")
print("==============================")

print("MAE  :", round(mae, 2))
print("MSE  :", round(mse, 2))
print("RMSE :", round(rmse, 2))
print("R2   :", round(r2, 4))


# =========================================================
# 14. FEATURE IMPORTANCE
# =========================================================

feature_importance = pd.DataFrame({

    "Feature": X.columns,

    "Importance": model.feature_importances_

})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)


print("\n==============================")
print("FEATURE IMPORTANCE")
print("==============================")

print(
    feature_importance.to_string(index=False)
)


# =========================================================
# 15. SAVE MODEL
# =========================================================

joblib.dump(
    {
        "model": model,
        "encoders": encoders,
        "features": X.columns.tolist()
    },
    "employee_absenteeism_random_forest.pkl"
)

print("\nModel saved successfully!")

print(
    "File: employee_absenteeism_random_forest.pkl"
)


# =========================================================
# 16. ACTUAL VS PREDICTED
# =========================================================

results = pd.DataFrame({

    "Actual Absenteeism Days": y_test.values,

    "Predicted Absenteeism Days":
        np.round(y_pred, 2)

})


print("\n==============================")
print("ACTUAL vs PREDICTED")
print("==============================")

print(
    results.head(10).to_string(index=False)
)


# =========================================================
# 17. SAMPLE PREDICTION
# =========================================================

print("\n==============================")
print("SAMPLE PREDICTION")
print("==============================")

sample = X_test.iloc[[0]]

sample_prediction = model.predict(sample)[0]

print(
    "Predicted Absenteeism Days:",
    round(sample_prediction, 2)
)