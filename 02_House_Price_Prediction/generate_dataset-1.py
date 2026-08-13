"""
generate_dataset.py
--------------------
Generates a realistic synthetic housing dataset and saves it as
'housing_dataset.csv'.

NOTE: This environment has no internet access, so the classic Boston
Housing dataset (also deprecated/removed from scikit-learn for ethical
reasons) or fetch_california_housing (requires a network download)
could not be used. Per the guideline's own allowance -- "Boston Housing
dataset OR ANY BASIC HOUSING DATASET with numeric features" -- this
script builds a synthetic dataset with a realistic underlying price
formula (size, location, bedrooms, age, distance to city center) plus
random noise and a few intentionally missing values, so every later
step (missing-data handling, normalization, regression, MSE) applies
exactly as it would to a real dataset.
"""

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 600

locations = ["Downtown", "Suburb", "Rural", "Uptown"]
location_price_factor = {"Downtown": 1.6, "Uptown": 1.3, "Suburb": 1.0, "Rural": 0.7}

size_sqft = rng.normal(1800, 600, n).clip(400, 5000)
bedrooms = rng.integers(1, 6, n)
bathrooms = rng.integers(1, 4, n)
age_years = rng.integers(0, 60, n)
distance_to_city_km = rng.exponential(8, n).clip(0.5, 50)
location = rng.choice(locations, n, p=[0.25, 0.35, 0.15, 0.25])

base_price = (
    size_sqft * 120
    + bedrooms * 8000
    + bathrooms * 5000
    - age_years * 600
    - distance_to_city_km * 900
)
loc_factor = np.array([location_price_factor[l] for l in location])
price = base_price * loc_factor
price += rng.normal(0, 15000, n)   # noise
price = price.clip(30000, None)

df = pd.DataFrame({
    "size_sqft": size_sqft.round(1),
    "bedrooms": bedrooms,
    "bathrooms": bathrooms,
    "age_years": age_years,
    "distance_to_city_km": distance_to_city_km.round(2),
    "location": location,
    "price": price.round(0),
})

# Intentionally introduce some missing values (as real datasets often have)
missing_idx = rng.choice(df.index, size=25, replace=False)
df.loc[missing_idx, "bathrooms"] = np.nan
missing_idx2 = rng.choice(df.index, size=15, replace=False)
df.loc[missing_idx2, "age_years"] = np.nan

df.to_csv("housing_dataset.csv", index=False)
print(f"Dataset created: {df.shape[0]} rows, {df.shape[1]} columns")
print(df.head())
print("\nMissing values:\n", df.isna().sum())
