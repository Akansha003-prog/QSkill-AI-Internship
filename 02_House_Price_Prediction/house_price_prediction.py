"""
House Price Prediction
=======================
Objective: Predict the price of a house based on features such as
size, location, and number of bedrooms.

Steps followed (per project guideline):
 1. Load the dataset and explore data distributions.
 2. Handle missing data and preprocess inputs (normalization).
 3. Split into train/test sets.
 4. Train a regression model (Linear Regression is classic for beginners).
 5. Evaluate predictions using metrics like Mean Squared Error (MSE).

Skills gained: Handling tabular data, regression, feature engineering,
basic metrics.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# -------------------------------------------------------------------
# 1. LOAD THE DATASET AND EXPLORE DATA DISTRIBUTIONS
# -------------------------------------------------------------------
df = pd.read_csv("housing_dataset.csv")
print("Dataset shape:", df.shape)
print(df.head(), "\n")
print(df.describe(include="all"), "\n")
print("Missing values per column:\n", df.isna().sum(), "\n")

numeric_cols = ["size_sqft", "bedrooms", "bathrooms", "age_years", "distance_to_city_km"]
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()
for i, col in enumerate(numeric_cols + ["price"]):
    axes[i].hist(df[col].dropna(), bins=25, color="#4C72B0", edgecolor="white")
    axes[i].set_title(f"Distribution: {col}")
plt.tight_layout()
plt.savefig("data_distributions.png", dpi=150)
print("Saved chart: data_distributions.png\n")

# -------------------------------------------------------------------
# 2. HANDLE MISSING DATA AND PREPROCESS INPUTS (NORMALIZATION)
# -------------------------------------------------------------------
X = df.drop(columns=["price"])
y = df["price"]

numeric_features = ["size_sqft", "bedrooms", "bathrooms", "age_years", "distance_to_city_km"]
categorical_features = ["location"]

numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),   # handle missing data
    ("scaler", StandardScaler()),                      # normalization
])
categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features),
])

# -------------------------------------------------------------------
# 3. SPLIT INTO TRAIN / TEST SETS
# -------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}\n")

# -------------------------------------------------------------------
# 4. TRAIN A REGRESSION MODEL (LINEAR REGRESSION)
# -------------------------------------------------------------------
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression()),
])
model.fit(X_train, y_train)

# -------------------------------------------------------------------
# 5. EVALUATE PREDICTIONS (MSE and other metrics)
# -------------------------------------------------------------------
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("=" * 50)
print("MODEL EVALUATION - Linear Regression")
print("=" * 50)
print(f"Mean Squared Error (MSE) : {mse:,.2f}")
print(f"Root MSE (RMSE)          : {rmse:,.2f}")
print(f"Mean Absolute Error (MAE): {mae:,.2f}")
print(f"R^2 Score                : {r2:.4f}")

# Feature importance via coefficients
feature_names = (
    numeric_features
    + list(model.named_steps["preprocessor"]
           .named_transformers_["cat"]
           .named_steps["onehot"]
           .get_feature_names_out(categorical_features))
)
coefs = model.named_steps["regressor"].coef_
coef_df = pd.DataFrame({"feature": feature_names, "coefficient": coefs})
coef_df = coef_df.sort_values("coefficient", key=abs, ascending=False)
print("\nFeature coefficients (standardized numeric features):")
print(coef_df.to_string(index=False))

# -------------------------------------------------------------------
# VISUALIZATIONS
# -------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(y_test, y_pred, alpha=0.5, color="#4C72B0")
lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
axes[0].plot(lims, lims, "r--", label="Perfect prediction")
axes[0].set_xlabel("Actual Price")
axes[0].set_ylabel("Predicted Price")
axes[0].set_title("Actual vs Predicted Price")
axes[0].legend()

axes[1].barh(coef_df["feature"], coef_df["coefficient"], color="#55A868")
axes[1].set_title("Feature Coefficients")
axes[1].set_xlabel("Coefficient value")

plt.tight_layout()
plt.savefig("prediction_results.png", dpi=150)
print("\nSaved chart: prediction_results.png")

# -------------------------------------------------------------------
# TRY IT ON A NEW HOUSE
# -------------------------------------------------------------------
new_house = pd.DataFrame([{
    "size_sqft": 2200,
    "bedrooms": 3,
    "bathrooms": 2,
    "age_years": 10,
    "distance_to_city_km": 4.5,
    "location": "Suburb",
}])
predicted_price = model.predict(new_house)[0]
print("\n" + "=" * 50)
print("PREDICTION ON A NEW HOUSE")
print("=" * 50)
print(new_house.to_string(index=False))
print(f"\nPredicted Price: ${predicted_price:,.0f}")
