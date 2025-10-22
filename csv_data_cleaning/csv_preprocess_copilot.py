"""
csv_pipeline_checklist.py

Exhaustive checklist for handling a CSV file in a pandas DataFrame.
Covers: 
1. Data Cleaning
2. Handling Missing Values
3. Handling Missing Values using Residuals
4. Feature Engineering
5. Variable Transformation
6. Feature Coding
7. Feature Scaling
8. Extended Variable Transformation
9. Visualization Tasks
10. Handling Outliers
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder, PowerTransformer
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression

# ============================================================
# 1. Load Data
# ============================================================
df = pd.read_csv("your_file.csv")

print("Initial Shape:", df.shape)
print("Initial Info:")
print(df.info())
print(df.head())

# ============================================================
# 2. Data Cleaning
# ============================================================
df = df.drop_duplicates()
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# Example: fix categorical typos
# df['category'] = df['category'].replace({'catgory':'category'})

# ============================================================
# 3. Handling Missing Values
# ============================================================
print("Missing Values:\n", df.isnull().sum())

# Simple imputation
imputer = SimpleImputer(strategy="mean")
# df['numeric_col'] = imputer.fit_transform(df[['numeric_col']])

# ============================================================
# 4. Handling Missing Values using Residuals
# ============================================================
def impute_with_residuals(df, target, features):
    model = LinearRegression()
    train = df[df[target].notnull()]
    test = df[df[target].isnull()]
    model.fit(train[features], train[target])
    preds = model.predict(test[features])
    df.loc[df[target].isnull(), target] = preds
    return df

# Example usage:
# df = impute_with_residuals(df, target="numeric_col", features=["feature1","feature2"])

# ============================================================
# 5. Feature Engineering
# ============================================================
# df['feature1_x_feature2'] = df['feature1'] * df['feature2']
# df['year'] = pd.to_datetime(df['date']).dt.year
# df['age_group'] = pd.cut(df['age'], bins=[0,18,35,50,100], labels=["child","young","adult","senior"])

# ============================================================
# 6. Variable Transformation
# ============================================================
# df['log_income'] = np.log1p(df['income'])
# pt = PowerTransformer(method='yeo-johnson')
# df['transformed'] = pt.fit_transform(df[['numeric_col']])

# ============================================================
# 7. Feature Coding
# ============================================================
# le = LabelEncoder()
# df['encoded_col'] = le.fit_transform(df['categorical_col'])

# ohe = OneHotEncoder(sparse=False, drop='first')
# encoded = ohe.fit_transform(df[['categorical_col']])
# df[ohe.get_feature_names_out(['categorical_col'])] = encoded

# ============================================================
# 8. Feature Scaling
# ============================================================
scaler = StandardScaler()
# df['scaled_col'] = scaler.fit_transform(df[['numeric_col']])

minmax = MinMaxScaler()
# df['minmax_col'] = minmax.fit_transform(df[['numeric_col']])

# ============================================================
# 9. Visualization Tasks
# ============================================================
# Distribution plots
# sns.histplot(df['numeric_col'], kde=True)
# plt.show()

# Boxplot for outliers
# sns.boxplot(x=df['numeric_col'])
# plt.show()

# Correlation heatmap
# plt.figure(figsize=(10,8))
# sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
# plt.show()

# Pairplot for relationships
# sns.pairplot(df.select_dtypes(include=np.number))
# plt.show()

# ============================================================
# 10. Handling Outliers
# ============================================================
def detect_outliers_iqr(series):
    Q1, Q3 = series.quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
    return series[(series < lower) | (series > upper)]

# Example usage:
# outliers = detect_outliers_iqr(df['numeric_col'])
# print("Outliers detected:", outliers)

# Strategies:
# - Remove outliers
# df = df[~df['numeric_col'].isin(outliers)]
# - Cap outliers (winsorization)
# df['numeric_col'] = np.where(df['numeric_col'] > upper, upper,
#                       np.where(df['numeric_col'] < lower, lower, df['numeric_col']))
# - Transform skewed data
# df['numeric_col_log'] = np.log1p(df['numeric_col'])

# ============================================================
# Final Checks
# ============================================================
print("Final Shape:", df.shape)
print("Final Preview:")
print(df.head())
