"""
Task 02 - Data Cleaning & Feature Engineering
Devixo Solutions

Dataset : California Housing (20,640 records, 10 columns)
Tools   : Python, Pandas, NumPy, Scikit-learn
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

pd.set_option("display.width", 150)
pd.set_option("display.max_columns", 30)

URL = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv"


# ------------------------------------------------------------------
# 1. Load the data
# ------------------------------------------------------------------
df = pd.read_csv(URL)
print("Rows, columns:", df.shape)
print(df.head())

# keep a copy of the raw data so we can compare later
raw = df.copy()


# ------------------------------------------------------------------
# 2. Statistics BEFORE preprocessing
# ------------------------------------------------------------------
print("\n===== BEFORE PREPROCESSING =====")
print(df.info())
print("\nMissing values:\n", df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())
before_stats = df.describe().T
print(before_stats)


# ------------------------------------------------------------------
# 3. Step by step cleaning (with reasons)
# ------------------------------------------------------------------

# --- 3.1 Duplicates ---
# Duplicate rows count the same house twice and can bias the model,
# so we drop them.
dups = df.duplicated().sum()
df = df.drop_duplicates().reset_index(drop=True)
print(f"\nDuplicates removed: {dups}")

# --- 3.2 Wrong data types ---
# ocean_proximity is text with only 5 values -> category (saves memory,
# and shows it is not a free text column).
# These columns are counts of people / houses / age in years, so they
# cannot have decimals -> integer.
df["ocean_proximity"] = df["ocean_proximity"].astype("category")
df["housing_median_age"] = df["housing_median_age"].round().astype(int)
df["population"] = df["population"].round().astype(int)
df["households"] = df["households"].round().astype(int)
print("\nData types after fixing:\n", df.dtypes)

# --- 3.3 Missing values ---
# Only total_bedrooms has missing values (about 1%). Deleting rows would
# waste data, so we fill with the median. Median is used because
# total_bedrooms is right skewed and the mean would be pulled by big values.
missing_before = df["total_bedrooms"].isnull().sum()
df["total_bedrooms"] = df["total_bedrooms"].fillna(df["total_bedrooms"].median())
print(f"\nMissing total_bedrooms filled: {missing_before}")

# --- 3.4 Outliers (IQR method) ---
# Some districts have huge room / population numbers (big blocks of flats).
# We do not delete them because they are real data. Instead we cap them at
# Q1 - 1.5*IQR and Q3 + 1.5*IQR so they stop dominating the scaling.
# median_house_value is our target, so we leave it alone.
outlier_cols = ["total_rooms", "total_bedrooms", "population",
                "households", "median_income"]

print("\nOutliers found (IQR rule):")
for col in outlier_cols:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n = ((df[col] < low) | (df[col] > high)).sum()
    print(f"  {col:15s} {n:5d}")
    df[col] = df[col].clip(low, high)

# --- 3.5 Feature engineering (6 new features) ---
# Raw totals depend on how big the district is. Ratios are easier to compare.
df["rooms_per_household"] = df["total_rooms"] / df["households"]
df["bedroom_ratio"] = df["total_bedrooms"] / df["total_rooms"]
df["people_per_household"] = df["population"] / df["households"]
df["income_per_room"] = df["median_income"] / df["rooms_per_household"]

# Age group: old houses and new houses behave differently in price
df["age_group"] = pd.cut(df["housing_median_age"],
                         bins=[0, 15, 30, 60],
                         labels=["new", "mid", "old"]).astype(str)

# Rough distance from San Francisco (degrees -> km, 1 degree ~ 111 km)
sf_lat, sf_lon = 37.77, -122.42
df["dist_to_sf_km"] = np.sqrt((df["latitude"] - sf_lat) ** 2 +
                              (df["longitude"] - sf_lon) ** 2) * 111

new_features = ["rooms_per_household", "bedroom_ratio", "people_per_household",
                "income_per_room", "age_group", "dist_to_sf_km"]
print("\nNew features:", new_features)

# Ratios can blow up when households is very small (e.g. 1 household and
# 500 people), so we cap the new ratio columns with the same IQR rule.
ratio_cols = ["rooms_per_household", "bedroom_ratio",
              "people_per_household", "income_per_room"]
for col in ratio_cols:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    df[col] = df[col].clip(q1 - 1.5 * iqr, q3 + 1.5 * iqr)

# --- 3.6 Encoding + scaling are done inside the pipeline below ---


# ------------------------------------------------------------------
# 4. Same steps as an automated Scikit-learn pipeline (BONUS)
# ------------------------------------------------------------------
class MissingFiller(BaseEstimator, TransformerMixin):
    """Fill numbers with median and text columns with most common value."""

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.fill_ = {}
        for c in X.columns:
            if pd.api.types.is_numeric_dtype(X[c]):
                self.fill_[c] = X[c].median()
            else:
                self.fill_[c] = X[c].mode()[0]
        return self

    def transform(self, X):
        return pd.DataFrame(X).fillna(self.fill_)


class OutlierCapper(BaseEstimator, TransformerMixin):
    """Cap values using the IQR rule. Limits are learned from train data."""

    def __init__(self, cols):
        self.cols = cols

    def fit(self, X, y=None):
        self.limits_ = {}
        for c in self.cols:
            q1, q3 = X[c].quantile([0.25, 0.75])
            iqr = q3 - q1
            self.limits_[c] = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)
        return self

    def transform(self, X):
        X = X.copy()
        for c, (low, high) in self.limits_.items():
            X[c] = X[c].clip(low, high)
        return X


class FeatureAdder(BaseEstimator, TransformerMixin):
    """Create the new features."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["rooms_per_household"] = X["total_rooms"] / X["households"]
        X["bedroom_ratio"] = X["total_bedrooms"] / X["total_rooms"]
        X["people_per_household"] = X["population"] / X["households"]
        X["income_per_room"] = X["median_income"] / X["rooms_per_household"]
        X["age_group"] = pd.cut(X["housing_median_age"], bins=[0, 15, 30, 60],
                                labels=["new", "mid", "old"]).astype(str)
        X["dist_to_sf_km"] = np.sqrt((X["latitude"] - 37.77) ** 2 +
                                     (X["longitude"] + 122.42) ** 2) * 111
        return X


num_cols = ["longitude", "latitude", "housing_median_age", "total_rooms",
            "total_bedrooms", "population", "households", "median_income",
            "rooms_per_household", "bedroom_ratio", "people_per_household",
            "income_per_room", "dist_to_sf_km"]
cat_cols = ["ocean_proximity", "age_group"]

encode_and_scale = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
])

pipeline = Pipeline([
    ("fill_missing", MissingFiller()),
    ("cap_outliers", OutlierCapper(["total_rooms", "total_bedrooms", "population",
                                    "households", "median_income"])),
    ("new_features", FeatureAdder()),
    ("cap_ratios", OutlierCapper(["rooms_per_household", "bedroom_ratio",
                                  "people_per_household", "income_per_room"])),
    ("encode_scale", encode_and_scale),
])

# run on the raw data (after removing duplicates and fixing types)
X_raw = raw.drop_duplicates().reset_index(drop=True)
X_raw["ocean_proximity"] = X_raw["ocean_proximity"].astype(str)
y = X_raw["median_house_value"]
X_raw = X_raw.drop(columns="median_house_value")

X_ready = pipeline.fit_transform(X_raw)
feature_names = pipeline.named_steps["encode_scale"].get_feature_names_out()
X_ready = pd.DataFrame(X_ready, columns=feature_names)
print("\nPipeline output shape:", X_ready.shape)
print("Missing values in pipeline output:", int(X_ready.isnull().sum().sum()))


# ------------------------------------------------------------------
# 5. Statistics AFTER preprocessing
# ------------------------------------------------------------------
print("\n===== AFTER PREPROCESSING =====")
print("Missing values:", int(df.isnull().sum().sum()))
print("Duplicate rows:", df.duplicated().sum())
after_stats = df.describe().T
print(after_stats)

print("\nScaled data (pipeline output) - mean should be ~0, std ~1:")
print(X_ready[[f"num__{c}" for c in num_cols]].describe().T[["mean", "std", "min", "max"]])

# compare a few columns before / after
print("\nBefore vs After (max values):")
cmp = pd.DataFrame({
    "before_max": raw[outlier_cols].max(),
    "after_max": df[outlier_cols].max(),
})
print(cmp)


# ------------------------------------------------------------------
# 6. Save the results
# ------------------------------------------------------------------
df.to_csv("housing_cleaned.csv", index=False)
X_ready.assign(median_house_value=y).to_csv("housing_model_ready.csv", index=False)
before_stats.to_csv("stats_before.csv")
after_stats.to_csv("stats_after.csv")
print("\nSaved: housing_cleaned.csv, housing_model_ready.csv, stats_before.csv, stats_after.csv")
