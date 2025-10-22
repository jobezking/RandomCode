"""
EXHAUSTIVE CHECKLIST FOR CSV DATA PROCESSING IN PYTHON
======================================================
Author: Data Science Workflow Guide
Purpose: Comprehensive checklist of steps and commands for DataFrame operations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import (StandardScaler, MinMaxScaler, RobustScaler, 
                                   LabelEncoder, OneHotEncoder)
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# Load CSV file
df = pd.read_csv('your_file.csv')

# ============================================================================
# 1. DATA CLEANING
# ============================================================================

print("="*80)
print("1. DATA CLEANING")
print("="*80)

# 1.1 Initial Data Exploration
print("\n--- 1.1 Initial Data Exploration ---")
print(df.head())                          # First 5 rows
print(df.tail())                          # Last 5 rows
print(df.shape)                           # Dimensions (rows, columns)
print(df.info())                          # Data types and non-null counts
print(df.describe())                      # Statistical summary
print(df.describe(include='object'))      # Summary for categorical variables
print(df.columns.tolist())                # List all column names
print(df.dtypes)                          # Data types of each column

# 1.2 Check for Duplicates
print("\n--- 1.2 Check for Duplicates ---")
print(f"Total duplicates: {df.duplicated().sum()}")
print(df[df.duplicated(keep=False)])      # View all duplicate rows
df_clean = df.drop_duplicates()           # Remove duplicates
df_clean = df.drop_duplicates(subset=['column_name'])  # Remove based on specific column
df_clean = df.drop_duplicates(keep='first')  # Keep first occurrence
df_clean = df.drop_duplicates(keep='last')   # Keep last occurrence

# 1.3 Handle Whitespace and String Issues
print("\n--- 1.3 Handle Whitespace ---")
df['column'] = df['column'].str.strip()              # Remove leading/trailing spaces
df['column'] = df['column'].str.lower()              # Convert to lowercase
df['column'] = df['column'].str.upper()              # Convert to uppercase
df['column'] = df['column'].str.title()              # Title case
df['column'] = df['column'].str.replace('  ', ' ')   # Remove double spaces
df.columns = df.columns.str.strip()                  # Clean column names
df.columns = df.columns.str.replace(' ', '_')        # Replace spaces in column names

# 1.4 Fix Data Types
print("\n--- 1.4 Fix Data Types ---")
df['column'] = df['column'].astype(int)              # Convert to integer
df['column'] = df['column'].astype(float)            # Convert to float
df['column'] = df['column'].astype(str)              # Convert to string
df['column'] = pd.to_datetime(df['column'])          # Convert to datetime
df['column'] = pd.to_numeric(df['column'], errors='coerce')  # Convert with error handling
df['column'] = df['column'].astype('category')       # Convert to categorical

# 1.5 Handle Inconsistent Values
print("\n--- 1.5 Handle Inconsistent Values ---")
df['column'].replace({'old_value': 'new_value'}, inplace=True)
df['column'].replace(['val1', 'val2'], 'new_val', inplace=True)
df.replace({'column': {'old': 'new'}}, inplace=True)

# 1.6 Remove Unwanted Characters
print("\n--- 1.6 Remove Unwanted Characters ---")
df['column'] = df['column'].str.replace('[^a-zA-Z0-9]', '', regex=True)  # Keep alphanumeric only
df['column'] = df['column'].str.replace('$', '')     # Remove dollar signs
df['column'] = df['column'].str.replace(',', '')     # Remove commas
df['column'] = df['column'].str.replace('%', '')     # Remove percentage signs

# 1.7 Drop Unnecessary Columns
print("\n--- 1.7 Drop Unnecessary Columns ---")
df_clean = df.drop(['column1', 'column2'], axis=1)   # Drop specific columns
df_clean = df.drop(columns=['column1'])              # Alternative syntax
df_clean = df.loc[:, df.columns != 'column_name']    # Drop by condition

# 1.8 Rename Columns
print("\n--- 1.8 Rename Columns ---")
df.rename(columns={'old_name': 'new_name'}, inplace=True)
df.columns = ['col1', 'col2', 'col3']                # Rename all columns


# ============================================================================
# 2. HANDLING MISSING VALUES AND OUTLIERS
# ============================================================================

print("\n" + "="*80)
print("2. HANDLING MISSING VALUES AND OUTLIERS")
print("="*80)

# 2.1 Identify Missing Values
print("\n--- 2.1 Identify Missing Values ---")
print(df.isnull().sum())                             # Count nulls per column
print(df.isnull().sum() / len(df) * 100)             # Percentage of nulls
print(df.isna().sum())                               # Alternative (isna = isnull)
missing_data = df.isnull().sum().sort_values(ascending=False)
print(missing_data[missing_data > 0])

# Visualize missing data
import missingno as msno  # pip install missingno
msno.matrix(df)
msno.heatmap(df)

# 2.2 Drop Missing Values
print("\n--- 2.2 Drop Missing Values ---")
df_clean = df.dropna()                               # Drop all rows with any NaN
df_clean = df.dropna(axis=1)                         # Drop columns with any NaN
df_clean = df.dropna(subset=['column'])              # Drop rows where specific column is NaN
df_clean = df.dropna(thresh=5)                       # Keep rows with at least 5 non-NaN values
df_clean = df.dropna(how='all')                      # Drop only if all values are NaN

# 2.3 Fill Missing Values - Simple Methods
print("\n--- 2.3 Fill Missing Values - Simple Methods ---")
df['column'].fillna(0, inplace=True)                 # Fill with zero
df['column'].fillna(df['column'].mean(), inplace=True)    # Fill with mean
df['column'].fillna(df['column'].median(), inplace=True)  # Fill with median
df['column'].fillna(df['column'].mode()[0], inplace=True) # Fill with mode
df['column'].fillna(method='ffill', inplace=True)    # Forward fill
df['column'].fillna(method='bfill', inplace=True)    # Backward fill
df.fillna(df.mean(), inplace=True)                   # Fill all numeric columns with mean

# 2.4 Fill Missing Values - Advanced Methods
print("\n--- 2.4 Fill Missing Values - Advanced Methods ---")
# Interpolation
df['column'].interpolate(method='linear', inplace=True)
df['column'].interpolate(method='polynomial', order=2, inplace=True)
df['column'].interpolate(method='time', inplace=True)

# Group-based imputation
df['column'] = df.groupby('category')['column'].transform(lambda x: x.fillna(x.mean()))

# Using SimpleImputer from sklearn
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')  # 'mean', 'median', 'most_frequent', 'constant'
df[['col1', 'col2']] = imputer.fit_transform(df[['col1', 'col2']])

# KNN Imputation
from sklearn.impute import KNNImputer
knn_imputer = KNNImputer(n_neighbors=5)
df_imputed = pd.DataFrame(knn_imputer.fit_transform(df), columns=df.columns)

# 2.5 Identify Outliers - Statistical Methods
print("\n--- 2.5 Identify Outliers - Statistical Methods ---")
# Z-score method
z_scores = np.abs(stats.zscore(df['column']))
outliers_z = df[z_scores > 3]
print(f"Outliers using Z-score: {len(outliers_z)}")

# IQR method
Q1 = df['column'].quantile(0.25)
Q3 = df['column'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers_iqr = df[(df['column'] < lower_bound) | (df['column'] > upper_bound)]
print(f"Outliers using IQR: {len(outliers_iqr)}")

# Percentile method
lower_percentile = df['column'].quantile(0.01)
upper_percentile = df['column'].quantile(0.99)
outliers_percentile = df[(df['column'] < lower_percentile) | (df['column'] > upper_percentile)]

# 2.6 Visualize Outliers
print("\n--- 2.6 Visualize Outliers ---")
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
df['column'].hist(bins=50)
plt.title('Histogram')
plt.subplot(1, 3, 2)
df.boxplot(column='column')
plt.title('Boxplot')
plt.subplot(1, 3, 3)
df['column'].plot(kind='kde')
plt.title('KDE Plot')
plt.tight_layout()
plt.show()

# 2.7 Handle Outliers
print("\n--- 2.7 Handle Outliers ---")
# Remove outliers
df_no_outliers = df[(df['column'] >= lower_bound) & (df['column'] <= upper_bound)]

# Cap outliers (Winsorization)
df['column_capped'] = df['column'].clip(lower=lower_bound, upper=upper_bound)

# Transform outliers
from scipy.stats import mstats
df['column_winsorized'] = mstats.winsorize(df['column'], limits=[0.05, 0.05])

# Replace with mean/median
df.loc[df['column'] > upper_bound, 'column'] = df['column'].median()
df.loc[df['column'] < lower_bound, 'column'] = df['column'].median()


# ============================================================================
# 3. HANDLING MISSING VALUES AND OUTLIERS USING RESIDUALS
# ============================================================================

print("\n" + "="*80)
print("3. HANDLING MISSING VALUES AND OUTLIERS USING RESIDUALS")
print("="*80)

# 3.1 Calculate Residuals from Linear Regression
print("\n--- 3.1 Calculate Residuals ---")
# Assume we predict 'target' from 'feature1' and 'feature2'
X = df[['feature1', 'feature2']].dropna()
y = df.loc[X.index, 'target']
model = LinearRegression()
model.fit(X, y)
predictions = model.predict(X)
residuals = y - predictions
df.loc[X.index, 'residuals'] = residuals

# 3.2 Identify Outliers Using Residuals
print("\n--- 3.2 Identify Outliers Using Residuals ---")
# Standardized residuals
std_residuals = (residuals - residuals.mean()) / residuals.std()
outliers_residuals = df.loc[X.index][np.abs(std_residuals) > 3]
print(f"Outliers using residuals: {len(outliers_residuals)}")

# Cook's Distance
from scipy.stats import chi2
n = len(X)
p = X.shape[1]
leverage = (X * np.linalg.inv(X.T.dot(X)).dot(X.T)).sum(axis=1)
cooks_d = (residuals**2 / (p * residuals.var())) * (leverage / (1 - leverage)**2)
threshold = chi2.ppf(0.95, p)
outliers_cooks = df.loc[X.index][cooks_d > threshold]

# 3.3 Handle Outliers Based on Residuals
print("\n--- 3.3 Handle Outliers Based on Residuals ---")
# Remove outlier rows
df_clean = df.loc[X.index][np.abs(std_residuals) <= 3]

# Impute outliers using predicted values
df.loc[X.index, 'target_cleaned'] = df.loc[X.index, 'target'].copy()
outlier_mask = np.abs(std_residuals) > 3
df.loc[X.index[outlier_mask], 'target_cleaned'] = predictions[outlier_mask]

# 3.4 Use Residuals for Missing Value Imputation
print("\n--- 3.4 Use Residuals for Missing Value Imputation ---")
# Predict missing values using regression
missing_mask = df['target'].isnull()
if missing_mask.sum() > 0:
    X_missing = df.loc[missing_mask, ['feature1', 'feature2']].dropna()
    predictions_missing = model.predict(X_missing)
    df.loc[X_missing.index, 'target_imputed'] = predictions_missing


# ============================================================================
# 4. FEATURE ENGINEERING
# ============================================================================

print("\n" + "="*80)
print("4. FEATURE ENGINEERING")
print("="*80)

# 4.1 Create New Features - Mathematical Operations
print("\n--- 4.1 Mathematical Operations ---")
df['sum_feature'] = df['col1'] + df['col2']
df['diff_feature'] = df['col1'] - df['col2']
df['product_feature'] = df['col1'] * df['col2']
df['ratio_feature'] = df['col1'] / (df['col2'] + 1e-8)  # Add small value to avoid division by zero
df['squared_feature'] = df['col1'] ** 2
df['sqrt_feature'] = np.sqrt(df['col1'])
df['log_feature'] = np.log1p(df['col1'])  # log(1 + x)
df['exp_feature'] = np.exp(df['col1'])

# 4.2 Statistical Features
print("\n--- 4.2 Statistical Features ---")
df['rolling_mean'] = df['col1'].rolling(window=3).mean()
df['rolling_std'] = df['col1'].rolling(window=3).std()
df['rolling_min'] = df['col1'].rolling(window=3).min()
df['rolling_max'] = df['col1'].rolling(window=3).max()
df['cumsum'] = df['col1'].cumsum()
df['cumprod'] = df['col1'].cumprod()
df['pct_change'] = df['col1'].pct_change()

# 4.3 Date/Time Features
print("\n--- 4.3 Date/Time Features ---")
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['dayofweek'] = df['date'].dt.dayofweek
df['quarter'] = df['date'].dt.quarter
df['week'] = df['date'].dt.isocalendar().week
df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
df['days_since_start'] = (df['date'] - df['date'].min()).dt.days

# 4.4 Binning (Discretization)
print("\n--- 4.4 Binning ---")
# Equal-width binning
df['age_bin'] = pd.cut(df['age'], bins=5)
df['age_bin'] = pd.cut(df['age'], bins=[0, 18, 35, 50, 100], labels=['child', 'young', 'middle', 'senior'])

# Equal-frequency binning
df['income_bin'] = pd.qcut(df['income'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])

# 4.5 Interaction Features
print("\n--- 4.5 Interaction Features ---")
df['interaction_1_2'] = df['feature1'] * df['feature2']
df['interaction_1_3'] = df['feature1'] * df['feature3']

# Polynomial features
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(df[['feature1', 'feature2']])
poly_df = pd.DataFrame(poly_features, columns=poly.get_feature_names_out())

# 4.6 Aggregation Features
print("\n--- 4.6 Aggregation Features ---")
df['mean_by_group'] = df.groupby('category')['value'].transform('mean')
df['sum_by_group'] = df.groupby('category')['value'].transform('sum')
df['count_by_group'] = df.groupby('category')['value'].transform('count')
df['std_by_group'] = df.groupby('category')['value'].transform('std')
df['rank_by_group'] = df.groupby('category')['value'].transform('rank')

# 4.7 Text Features
print("\n--- 4.7 Text Features ---")
df['text_length'] = df['text_column'].str.len()
df['word_count'] = df['text_column'].str.split().str.len()
df['char_count'] = df['text_column'].str.replace(' ', '').str.len()
df['uppercase_count'] = df['text_column'].str.findall(r'[A-Z]').str.len()
df['digit_count'] = df['text_column'].str.findall(r'\d').str.len()

# 4.8 Domain-Specific Features
print("\n--- 4.8 Domain-Specific Features ---")
# Example: Customer Lifetime Value
df['customer_lifetime_value'] = df['avg_purchase'] * df['purchase_frequency'] * df['customer_lifespan']

# Example: BMI calculation
df['bmi'] = df['weight'] / (df['height'] ** 2)


# ============================================================================
# 5. VARIABLE TRANSFORMATION
# ============================================================================

print("\n" + "="*80)
print("5. VARIABLE TRANSFORMATION")
print("="*80)

# 5.1 Log Transformation
print("\n--- 5.1 Log Transformation ---")
df['log_transform'] = np.log(df['column'])
df['log1p_transform'] = np.log1p(df['column'])  # log(1 + x), handles zeros

# 5.2 Square Root Transformation
print("\n--- 5.2 Square Root Transformation ---")
df['sqrt_transform'] = np.sqrt(df['column'])

# 5.3 Box-Cox Transformation
print("\n--- 5.3 Box-Cox Transformation ---")
df['boxcox_transform'], lambda_param = stats.boxcox(df['column'] + 1)  # Add 1 if data has zeros
print(f"Optimal lambda: {lambda_param}")

# 5.4 Yeo-Johnson Transformation
print("\n--- 5.4 Yeo-Johnson Transformation ---")
from sklearn.preprocessing import PowerTransformer
pt = PowerTransformer(method='yeo-johnson')
df['yeojohnson_transform'] = pt.fit_transform(df[['column']])

# 5.5 Reciprocal Transformation
print("\n--- 5.5 Reciprocal Transformation ---")
df['reciprocal_transform'] = 1 / (df['column'] + 1e-8)

# 5.6 Exponential Transformation
print("\n--- 5.6 Exponential Transformation ---")
df['exp_transform'] = np.exp(df['column'])

# 5.7 Square Transformation
print("\n--- 5.7 Square Transformation ---")
df['square_transform'] = df['column'] ** 2

# 5.8 Cube Root Transformation
print("\n--- 5.8 Cube Root Transformation ---")
df['cbrt_transform'] = np.cbrt(df['column'])


# ============================================================================
# 6. FEATURE CODING (ENCODING)
# ============================================================================

print("\n" + "="*80)
print("6. FEATURE CODING (ENCODING)")
print("="*80)

# 6.1 Label Encoding (Ordinal)
print("\n--- 6.1 Label Encoding ---")
le = LabelEncoder()
df['category_encoded'] = le.fit_transform(df['category'])
print(f"Classes: {le.classes_}")

# Manual label encoding with mapping
mapping = {'low': 0, 'medium': 1, 'high': 2}
df['priority_encoded'] = df['priority'].map(mapping)

# 6.2 One-Hot Encoding (Nominal)
print("\n--- 6.2 One-Hot Encoding ---")
df_onehot = pd.get_dummies(df, columns=['category'], prefix='cat')
df_onehot = pd.get_dummies(df['category'], prefix='cat', drop_first=True)  # Drop first to avoid multicollinearity

# Using sklearn
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse_output=False, drop='first')
encoded = ohe.fit_transform(df[['category']])
df_encoded = pd.DataFrame(encoded, columns=ohe.get_feature_names_out())

# 6.3 Ordinal Encoding
print("\n--- 6.3 Ordinal Encoding ---")
from sklearn.preprocessing import OrdinalEncoder
oe = OrdinalEncoder(categories=[['low', 'medium', 'high']])
df['priority_ordinal'] = oe.fit_transform(df[['priority']])

# 6.4 Target Encoding (Mean Encoding)
print("\n--- 6.4 Target Encoding ---")
target_mean = df.groupby('category')['target'].mean()
df['category_target_encoded'] = df['category'].map(target_mean)

# 6.5 Frequency Encoding
print("\n--- 6.5 Frequency Encoding ---")
freq = df['category'].value_counts()
df['category_freq_encoded'] = df['category'].map(freq)

# 6.6 Binary Encoding
print("\n--- 6.6 Binary Encoding ---")
# Using category_encoders library
try:
    import category_encoders as ce
    be = ce.BinaryEncoder(cols=['category'])
    df_binary = be.fit_transform(df)
except ImportError:
    print("Install category_encoders: pip install category-encoders")

# 6.7 Hash Encoding
print("\n--- 6.7 Hash Encoding ---")
from sklearn.feature_extraction import FeatureHasher
h = FeatureHasher(n_features=10, input_type='string')
hashed = h.transform(df['category'].astype(str))
df_hashed = pd.DataFrame(hashed.toarray())

# 6.8 Leave-One-Out Encoding
print("\n--- 6.8 Leave-One-Out Encoding ---")
try:
    import category_encoders as ce
    loo = ce.LeaveOneOutEncoder(cols=['category'])
    df_loo = loo.fit_transform(df[['category']], df['target'])
except ImportError:
    print("Install category_encoders: pip install category-encoders")


# ============================================================================
# 7. FEATURE SCALING
# ============================================================================

print("\n" + "="*80)
print("7. FEATURE SCALING")
print("="*80)

# 7.1 Standardization (Z-score Normalization)
print("\n--- 7.1 Standardization ---")
scaler = StandardScaler()
df['feature_standardized'] = scaler.fit_transform(df[['feature']])
# For multiple columns
df[['col1_std', 'col2_std']] = scaler.fit_transform(df[['col1', 'col2']])

# Manual standardization
df['feature_std_manual'] = (df['feature'] - df['feature'].mean()) / df['feature'].std()

# 7.2 Min-Max Normalization
print("\n--- 7.2 Min-Max Normalization ---")
minmax = MinMaxScaler()
df['feature_minmax'] = minmax.fit_transform(df[['feature']])
# Custom range
minmax_custom = MinMaxScaler(feature_range=(0, 10))
df['feature_minmax_custom'] = minmax_custom.fit_transform(df[['feature']])

# Manual min-max
df['feature_minmax_manual'] = (df['feature'] - df['feature'].min()) / (df['feature'].max() - df['feature'].min())

# 7.3 Robust Scaling
print("\n--- 7.3 Robust Scaling ---")
robust = RobustScaler()
df['feature_robust'] = robust.fit_transform(df[['feature']])

# Manual robust scaling
median = df['feature'].median()
q1 = df['feature'].quantile(0.25)
q3 = df['feature'].quantile(0.75)
iqr = q3 - q1
df['feature_robust_manual'] = (df['feature'] - median) / iqr

# 7.4 MaxAbs Scaling
print("\n--- 7.4 MaxAbs Scaling ---")
from sklearn.preprocessing import MaxAbsScaler
maxabs = MaxAbsScaler()
df['feature_maxabs'] = maxabs.fit_transform(df[['feature']])

# 7.5 Normalizer (L1, L2)
print("\n--- 7.5 Normalizer ---")
from sklearn.preprocessing import Normalizer
# L2 normalization (Euclidean norm)
normalizer_l2 = Normalizer(norm='l2')
df_normalized_l2 = normalizer_l2.fit_transform(df[['col1', 'col2']])

# L1 normalization (Manhattan norm)
normalizer_l1 = Normalizer(norm='l1')
df_normalized_l1 = normalizer_l1.fit_transform(df[['col1', 'col2']])

# 7.6 Quantile Transformation
print("\n--- 7.6 Quantile Transformation ---")
from sklearn.preprocessing import QuantileTransformer
qt = QuantileTransformer(output_distribution='uniform')
df['feature_quantile_uniform'] = qt.fit_transform(df[['feature']])

qt_normal = QuantileTransformer(output_distribution='normal')
df['feature_quantile_normal'] = qt_normal.fit_transform(df[['feature']])

# 7.7 Power Transformation
print("\n--- 7.7 Power Transformation ---")
pt = PowerTransformer(method='box-cox')  # or 'yeo-johnson'
df['feature_power'] = pt.fit_transform(df[['feature']] + 1)  # Add 1 if data has zeros


# ============================================================================
# 8. VARIABLE TRANSFORMATION (Additional Methods)
# ============================================================================

print("\n" + "="*80)
print("8. ADDITIONAL VARIABLE TRANSFORMATION")
print("="*80)

# 8.1 Rank Transformation
print("\n--- 8.1 Rank Transformation ---")
df['feature_rank'] = df['feature'].rank()
df['feature_rank_pct'] = df['feature'].rank(pct=True)

# 8.2 Sigmoid Transformation
print("\n--- 8.2 Sigmoid Transformation ---")
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
df['feature_sigmoid'] = sigmoid(df['feature'])

# 8.3 Tanh Transformation
print("\n--- 8.3 Tanh Transformation ---")
df['feature_tanh'] = np.tanh(df['feature'])

# 8.4 Arcsinh Transformation
print("\n--- 8.4 Arcsinh Transformation ---")
df['feature_arcsinh'] = np.arcsinh(df['feature'])

# 8.5 Johnson Transformation
print("\n--- 8.5 Johnson Transformation ---")
# Requires finding optimal parameters
from scipy import optimize

def johnson_su(x, gamma, delta, xi, lambda_):
    z = gamma + delta * np.arcsinh((x - xi) / lambda_)
    return z

# 8.6 Trimming (Capping)
print("\n--- 8.6 Trimming/Capping ---")
lower_cap = df['feature'].quantile(0.01)
upper_cap = df['feature'].quantile(0.99)
df['feature_capped'] = df['feature'].clip(lower=lower_cap, upper=upper_cap)

# 8.7 Binomial Transformation
print("\n--- 8.7 Binomial Transformation ---")
df['feature_binary'] = (df['feature'] > df['feature'].median()).astype(int)

# 8.8 Custom Transformation Function
print("\n--- 8.8 Custom Transformation ---")
from sklearn.preprocessing import FunctionTransformer

def custom_transform(x):
    return np.log1p(x ** 2)

transformer = FunctionTransformer(custom_transform)
df['feature_custom'] = transformer.fit_transform(df[['feature']])


# ============================================================================
# 9. VISUALIZATION TASKS
# ============================================================================

print("\n" + "="*80)
print("9. VISUALIZATION TASKS")
print("="*80)

# 9.1 Univariate Analysis - Distribution Plots
print("\n--- 9.1 Univariate Analysis ---")

# Histogram
plt.figure(figsize=(10, 6))
df['column'].hist(bins=30, edgecolor='black')
plt.title('Histogram of Column')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()

# Multiple histograms
df.hist(figsize=(15, 10), bins=30)
plt.tight_layout()
plt.show()

# Density plot (KDE)
plt.figure(figsize=(10, 6))
df['column'].plot(kind='kde')
plt.title('Density Plot')
plt.xlabel('Value')
plt.show()

# Box plot (single variable)
plt.figure(figsize=(8, 6))
df.boxplot(column='column')
plt.title('Box Plot')
plt.ylabel('Value')
plt.show()

# Violin plot
plt.figure(figsize=(10, 6))
sns.violinplot(data=df, y='column')
plt.title('Violin Plot')
plt.show()

# Bar plot for categorical data
plt.figure(figsize=(10, 6))
df['category'].value_counts().plot(kind='bar')
plt.title('Category Distribution')
plt.xlabel('Category')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# Pie chart
plt.figure(figsize=(8, 8))
df['category'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title('Category Distribution')
plt.ylabel('')
plt.show()

# Count plot (seaborn)
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='category')
plt.title('Count Plot')
plt.xticks(rotation=45)
plt.show()

# 9.2 Bivariate Analysis
print("\n--- 9.2 Bivariate Analysis ---")

# Scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(df['feature1'], df['feature2'], alpha=0.5)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Scatter Plot')
plt.show()

# Scatter plot with regression line
plt.figure(figsize=(10, 6))
sns.regplot(data=df, x='feature1', y='feature2')
plt.title('Scatter Plot with Regression Line')
plt.show()

# Scatter plot colored by category
plt.figure(figsize=(10, 6))
for category in df['category'].unique():
    subset = df[df['category'] == category]
    plt.scatter(subset['feature1'], subset['feature2'], label=category, alpha=0.6)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Scatter Plot by Category')
plt.legend()
plt.show()

# Line plot (time series)
plt.figure(figsize=(12, 6))
df.plot(x='date', y='value')
plt.title('Time Series Plot')
plt.xlabel('Date')
plt.ylabel('Value')
plt.show()

# Box plot by category
plt.figure(figsize=(10, 6))
df.boxplot(column='value', by='category')
plt.title('Box Plot by Category')
plt.suptitle('')
plt.show()

# Violin plot by category
plt.figure(figsize=(12, 6))
sns.violinplot(data=df, x='category', y='value')
plt.title('Violin Plot by Category')
plt.xticks(rotation=45)
plt.show()

# Swarm plot
plt.figure(figsize=(12, 6))
sns.swarmplot(data=df, x='category', y='value')
plt.title('Swarm Plot')
plt.xticks(rotation=45)
plt.show()

# Joint plot (scatter + distributions)
sns.jointplot(data=df, x='feature1', y='feature2', kind='scatter')
plt.show()

# Hexbin plot (for dense data)
plt.figure(figsize=(10, 6))
plt.hexbin(df['feature1'], df['feature2'], gridsize=20, cmap='Blues')
plt.colorbar(label='Count')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Hexbin Plot')
plt.show()

# 9.3 Multivariate Analysis
print("\n--- 9.3 Multivariate Analysis ---")

# Correlation heatmap
plt.figure(figsize=(12, 10))
corr_matrix = df.select_dtypes(include=[np.number]).corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=1, fmt='.2f')
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()

# Pair plot
sns.pairplot(df[['feature1', 'feature2', 'feature3', 'target']])
plt.show()

# Pair plot with hue
sns.pairplot(df, hue='category', vars=['feature1', 'feature2', 'feature3'])
plt.show()

# Parallel coordinates plot
from pandas.plotting import parallel_coordinates
plt.figure(figsize=(12, 6))
parallel_coordinates(df[['feature1', 'feature2', 'feature3', 'category']], 
                     'category', colormap='viridis')
plt.title('Parallel Coordinates Plot')
plt.legend(loc='upper right')
plt.show()

# 3D scatter plot
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df['feature1'], df['feature2'], df['feature3'], c=df['target'], cmap='viridis')
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Feature 3')
plt.title('3D Scatter Plot')
plt.show()

# Andrews curves
from pandas.plotting import andrews_curves
plt.figure(figsize=(12, 6))
andrews_curves(df[['feature1', 'feature2', 'feature3', 'category']], 'category')
plt.title('Andrews Curves')
plt.legend(loc='upper right')
plt.show()

# 9.4 Missing Data Visualization
print("\n--- 9.4 Missing Data Visualization ---")

# Missing data bar chart
missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
plt.figure(figsize=(10, 6))
missing.plot(kind='bar')
plt.title('Missing Values by Column')
plt.xlabel('Column')
plt.ylabel('Number of Missing Values')
plt.xticks(rotation=45)
plt.show()

# Missing data heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.isnull(), cbar=True, yticklabels=False, cmap='viridis')
plt.title('Missing Data Heatmap')
plt.show()

# Using missingno library (if installed)
try:
    import missingno as msno
    msno.matrix(df)
    plt.show()
    msno.bar(df)
    plt.show()
    msno.heatmap(df)
    plt.show()
    msno.dendrogram(df)
    plt.show()
except ImportError:
    print("Install missingno: pip install missingno")

# 9.5 Outlier Visualization
print("\n--- 9.5 Outlier Visualization ---")

# Box plots for all numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns
n_cols = len(numeric_cols)
n_rows = (n_cols + 2) // 3
fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5*n_rows))
axes = axes.flatten()
for i, col in enumerate(numeric_cols):
    axes[i].boxplot(df[col].dropna())
    axes[i].set_title(f'Box Plot: {col}')
    axes[i].set_ylabel('Value')
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])
plt.tight_layout()
plt.show()

# Z-score visualization
from scipy import stats
z_scores = np.abs(stats.zscore(df.select_dtypes(include=[np.number]).dropna()))
plt.figure(figsize=(12, 6))
plt.hist(z_scores.flatten(), bins=50, edgecolor='black')
plt.axvline(x=3, color='r', linestyle='--', label='Threshold (3)')
plt.title('Distribution of Z-scores')
plt.xlabel('Z-score')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# Scatter plot highlighting outliers (IQR method)
Q1 = df['column'].quantile(0.25)
Q3 = df['column'].quantile(0.75)
IQR = Q3 - Q1
outlier_mask = (df['column'] < Q1 - 1.5*IQR) | (df['column'] > Q3 + 1.5*IQR)
plt.figure(figsize=(10, 6))
plt.scatter(df.index[~outlier_mask], df.loc[~outlier_mask, 'column'], 
            label='Normal', alpha=0.6)
plt.scatter(df.index[outlier_mask], df.loc[outlier_mask, 'column'], 
            color='red', label='Outliers', alpha=0.8)
plt.title('Outlier Detection (IQR Method)')
plt.xlabel('Index')
plt.ylabel('Value')
plt.legend()
plt.show()

# 9.6 Distribution Analysis
print("\n--- 9.6 Distribution Analysis ---")

# Q-Q plot (Quantile-Quantile)
from scipy import stats
plt.figure(figsize=(10, 6))
stats.probplot(df['column'].dropna(), dist="norm", plot=plt)
plt.title('Q-Q Plot')
plt.show()

# Histogram with normal curve overlay
plt.figure(figsize=(10, 6))
mu, std = df['column'].mean(), df['column'].std()
count, bins, ignored = plt.hist(df['column'].dropna(), bins=30, density=True, alpha=0.6)
plt.plot(bins, 1/(std * np.sqrt(2 * np.pi)) * 
         np.exp( - (bins - mu)**2 / (2 * std**2) ),
         linewidth=2, color='r')
plt.title('Histogram with Normal Distribution')
plt.xlabel('Value')
plt.ylabel('Density')
plt.show()

# 9.7 Time Series Visualizations
print("\n--- 9.7 Time Series Visualizations ---")

# Time series with rolling mean
plt.figure(figsize=(14, 6))
plt.plot(df['date'], df['value'], label='Original', alpha=0.5)
plt.plot(df['date'], df['value'].rolling(window=7).mean(), 
         label='7-day Rolling Mean', linewidth=2)
plt.title('Time Series with Rolling Mean')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.show()

# Seasonal decomposition
from statsmodels.tsa.seasonal import seasonal_decompose
result = seasonal_decompose(df.set_index('date')['value'], 
                           model='additive', period=30)
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(14, 10))
result.observed.plot(ax=ax1)
ax1.set_ylabel('Observed')
result.trend.plot(ax=ax2)
ax2.set_ylabel('Trend')
result.seasonal.plot(ax=ax3)
ax3.set_ylabel('Seasonal')
result.resid.plot(ax=ax4)
ax4.set_ylabel('Residual')
plt.tight_layout()
plt.show()

# 9.8 Categorical Analysis
print("\n--- 9.8 Categorical Analysis ---")

# Stacked bar chart
pd.crosstab(df['category1'], df['category2']).plot(kind='bar', stacked=True, 
                                                     figsize=(10, 6))
plt.title('Stacked Bar Chart')
plt.xlabel('Category 1')
plt.ylabel('Count')
plt.legend(title='Category 2')
plt.show()

# Grouped bar chart
pd.crosstab(df['category1'], df['category2']).plot(kind='bar', figsize=(10, 6))
plt.title('Grouped Bar Chart')
plt.xlabel('Category 1')
plt.ylabel('Count')
plt.legend(title='Category 2')
plt.show()

# Heatmap for categorical relationships
plt.figure(figsize=(10, 8))
crosstab = pd.crosstab(df['category1'], df['category2'])
sns.heatmap(crosstab, annot=True, fmt='d', cmap='YlOrRd')
plt.title('Categorical Relationship Heatmap')
plt.show()

# 9.9 Feature Importance Visualization
print("\n--- 9.9 Feature Importance Visualization ---")

# Feature correlation with target
correlations = df.corr()['target'].drop('target').sort_values(ascending=False)
plt.figure(figsize=(10, 8))
correlations.plot(kind='barh')
plt.title('Feature Correlation with Target')
plt.xlabel('Correlation Coefficient')
plt.show()

# 9.10 Advanced Visualizations
print("\n--- 9.10 Advanced Visualizations ---")

# Ridge plot (joy plot)
try:
    import joypy
    fig, axes = joypy.joyplot(df, column=['feature1', 'feature2', 'feature3'])
    plt.show()
except ImportError:
    print("Install joypy: pip install joypy")

# Radar chart
from math import pi
categories = ['Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5']
values = df[['feature1', 'feature2', 'feature3', 'feature4', 'feature5']].mean().values.tolist()
N = len(categories)
angles = [n / float(N) * 2 * pi for n in range(N)]
values += values[:1]
angles += angles[:1]
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
ax.plot(angles, values)
ax.fill(angles, values, alpha=0.25)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)
plt.title('Radar Chart')
plt.show()

# Bubble chart
plt.figure(figsize=(10, 6))
plt.scatter(df['feature1'], df['feature2'], s=df['feature3']*100, 
            alpha=0.5, c=df['target'], cmap='viridis')
plt.colorbar(label='Target')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Bubble Chart (size = Feature 3)')
plt.show()


# ============================================================================
# 10. HANDLING OUTLIERS (COMPREHENSIVE)
# ============================================================================

print("\n" + "="*80)
print("10. HANDLING OUTLIERS (COMPREHENSIVE)")
print("="*80)

# 10.1 Detection Methods
print("\n--- 10.1 Detection Methods ---")

# Z-Score Method
from scipy import stats
z_scores = np.abs(stats.zscore(df['column'].dropna()))
threshold = 3
outliers_zscore = df[z_scores > threshold]
print(f"Outliers detected (Z-score): {len(outliers_zscore)}")

# Modified Z-Score (using median)
median = df['column'].median()
mad = np.median(np.abs(df['column'] - median))
modified_z_scores = 0.6745 * (df['column'] - median) / mad
outliers_modified_z = df[np.abs(modified_z_scores) > 3.5]
print(f"Outliers detected (Modified Z-score): {len(outliers_modified_z)}")

# IQR Method
Q1 = df['column'].quantile(0.25)
Q3 = df['column'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers_iqr = df[(df['column'] < lower_bound) | (df['column'] > upper_bound)]
print(f"Outliers detected (IQR): {len(outliers_iqr)}")

# Percentile Method
lower_percentile = df['column'].quantile(0.01)
upper_percentile = df['column'].quantile(0.99)
outliers_percentile = df[(df['column'] < lower_percentile) | 
                         (df['column'] > upper_percentile)]
print(f"Outliers detected (Percentile): {len(outliers_percentile)}")

# Isolation Forest
from sklearn.ensemble import IsolationForest
iso_forest = IsolationForest(contamination=0.1, random_state=42)
outliers_iso = iso_forest.fit_predict(df[['column']])
df['outlier_iso'] = outliers_iso
outliers_iso_df = df[df['outlier_iso'] == -1]
print(f"Outliers detected (Isolation Forest): {len(outliers_iso_df)}")

# Local Outlier Factor (LOF)
from sklearn.neighbors import LocalOutlierFactor
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
outliers_lof = lof.fit_predict(df[['feature1', 'feature2']])
df['outlier_lof'] = outliers_lof
outliers_lof_df = df[df['outlier_lof'] == -1]
print(f"Outliers detected (LOF): {len(outliers_lof_df)}")

# DBSCAN Clustering
from sklearn.cluster import DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
clusters = dbscan.fit_predict(df[['feature1', 'feature2']])
df['cluster'] = clusters
outliers_dbscan = df[df['cluster'] == -1]
print(f"Outliers detected (DBSCAN): {len(outliers_dbscan)}")

# One-Class SVM
from sklearn.svm import OneClassSVM
oc_svm = OneClassSVM(nu=0.1)
outliers_svm = oc_svm.fit_predict(df[['column']])
df['outlier_svm'] = outliers_svm
outliers_svm_df = df[df['outlier_svm'] == -1]
print(f"Outliers detected (One-Class SVM): {len(outliers_svm_df)}")

# Elliptic Envelope
from sklearn.covariance import EllipticEnvelope
ee = EllipticEnvelope(contamination=0.1)
outliers_ee = ee.fit_predict(df[['feature1', 'feature2']])
df['outlier_ee'] = outliers_ee
outliers_ee_df = df[df['outlier_ee'] == -1]
print(f"Outliers detected (Elliptic Envelope): {len(outliers_ee_df)}")

# Mahalanobis Distance
from scipy.spatial.distance import mahalanobis
features = df[['feature1', 'feature2']].dropna()
mean = features.mean().values
cov_matrix = features.cov().values
inv_cov = np.linalg.inv(cov_matrix)
mahal_dist = features.apply(lambda x: mahalanobis(x, mean, inv_cov), axis=1)
threshold_mahal = 3
outliers_mahal = df.loc[features.index][mahal_dist > threshold_mahal]
print(f"Outliers detected (Mahalanobis): {len(outliers_mahal)}")

# 10.2 Handling Strategies
print("\n--- 10.2 Handling Strategies ---")

# Strategy 1: Remove Outliers
df_no_outliers = df[(df['column'] >= lower_bound) & (df['column'] <= upper_bound)]
print(f"Original shape: {df.shape}, After removal: {df_no_outliers.shape}")

# Strategy 2: Cap/Clip Outliers (Winsorization)
df['column_capped'] = df['column'].clip(lower=lower_bound, upper=upper_bound)

# Winsorization with scipy
from scipy.stats.mstats import winsorize
df['column_winsorized'] = winsorize(df['column'], limits=[0.05, 0.05])

# Strategy 3: Replace with Mean/Median/Mode
outlier_mask = (df['column'] < lower_bound) | (df['column'] > upper_bound)
df.loc[outlier_mask, 'column_replaced'] = df['column'].median()

# Strategy 4: Transform Data
df['column_log'] = np.log1p(df['column'])  # Log transformation
df['column_sqrt'] = np.sqrt(df['column'])  # Square root transformation
df['column_boxcox'], _ = stats.boxcox(df['column'] + 1)  # Box-Cox

# Strategy 5: Binning
df['column_binned'] = pd.cut(df['column'], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])

# Strategy 6: Separate Treatment
df['is_outlier'] = outlier_mask.astype(int)  # Create binary indicator

# Strategy 7: Impute with Predicted Values
from sklearn.linear_model import LinearRegression
mask = ~outlier_mask
X_train = df.loc[mask, ['feature1', 'feature2']]
y_train = df.loc[mask, 'column']
model = LinearRegression()
model.fit(X_train, y_train)
X_outliers = df.loc[outlier_mask, ['feature1', 'feature2']]
df.loc[outlier_mask, 'column_imputed'] = model.predict(X_outliers)

# Strategy 8: Rank Transformation
df['column_rank'] = df['column'].rank(pct=True)

# Strategy 9: Quantile Transformation
from sklearn.preprocessing import QuantileTransformer
qt = QuantileTransformer(output_distribution='normal')
df['column_quantile'] = qt.fit_transform(df[['column']])

# Strategy 10: Robust Scaling (preserves outliers but scales data)
from sklearn.preprocessing import RobustScaler
robust_scaler = RobustScaler()
df['column_robust'] = robust_scaler.fit_transform(df[['column']])

# 10.3 Multivariate Outlier Detection
print("\n--- 10.3 Multivariate Outlier Detection ---")

# Cook's Distance
from sklearn.linear_model import LinearRegression
X = df[['feature1', 'feature2']].dropna()
y = df.loc[X.index, 'target']
model = LinearRegression()
model.fit(X, y)
predictions = model.predict(X)
residuals = y - predictions
mse = np.mean(residuals**2)
n = len(X)
p = X.shape[1]
leverage = (X * np.linalg.pinv(X.T.dot(X)).dot(X.T)).sum(axis=1)
cooks_d = (residuals**2 / (p * mse)) * (leverage / (1 - leverage)**2)
threshold_cooks = 4 / n
outliers_cooks = df.loc[X.index][cooks_d > threshold_cooks]
print(f"Outliers detected (Cook's Distance): {len(outliers_cooks)}")

# Leverage Points (Hat Values)
threshold_leverage = 2 * p / n
high_leverage = df.loc[X.index][leverage > threshold_leverage]
print(f"High leverage points: {len(high_leverage)}")

# DFFITS (Difference in Fits)
dffits = np.sqrt(leverage / (1 - leverage)) * (residuals / np.sqrt(mse * (1 - leverage)))
threshold_dffits = 2 * np.sqrt(p / n)
outliers_dffits = df.loc[X.index][np.abs(dffits) > threshold_dffits]
print(f"Outliers detected (DFFITS): {len(outliers_dffits)}")

# 10.4 Domain-Specific Outlier Handling
print("\n--- 10.4 Domain-Specific Outlier Handling ---")

# Business rule-based removal
df_filtered = df[(df['age'] >= 0) & (df['age'] <= 120)]  # Age constraints
df_filtered = df_filtered[(df_filtered['price'] > 0)]     # Price must be positive

# Seasonal outlier detection (for time series)
df['month'] = df['date'].dt.month
monthly_stats = df.groupby('month')['value'].agg(['mean', 'std'])
df = df.merge(monthly_stats, left_on='month', right_index=True, suffixes=('', '_monthly'))
z_seasonal = (df['value'] - df['mean_monthly']) / df['std_monthly']
seasonal_outliers = df[np.abs(z_seasonal) > 3]
print(f"Seasonal outliers: {len(seasonal_outliers)}")

# 10.5 Evaluation After Outlier Treatment
print("\n--- 10.5 Evaluation After Outlier Treatment ---")

# Compare statistics before and after
print("Before outlier treatment:")
print(df['column'].describe())
print("\nAfter outlier treatment:")
print(df_no_outliers['column'].describe())

# Visual comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.boxplot(df['column'].dropna())
ax1.set_title('Before Outlier Treatment')
ax2.boxplot(df_no_outliers['column'].dropna())
ax2.set_title('After Outlier Treatment')
plt.show()


# ============================================================================
# FINAL STEPS
# ============================================================================

print("\n" + "="*80)
print("FINAL STEPS")
print("="*80)

# Check final dataset
print("\n--- Final Dataset Info ---")
print(df.info())
print(df.head())
print(f"Shape: {df.shape}")

# Check for remaining missing values
print("\n--- Remaining Missing Values ---")
print(df.isnull().sum())

# Check for infinite values
print("\n--- Infinite Values Check ---")
print(np.isinf(df.select_dtypes(include=[np.number])).sum())

# Correlation matrix
print("\n--- Correlation Matrix ---")
corr_matrix = df.select_dtypes(include=[np.number]).corr()
print(corr_matrix)

# Save processed data
df.to_csv('processed_data.csv', index=False)
print("\n✓ Processed data saved to 'processed_data.csv'")

print("\n" + "="*80)
print("DATA PROCESSING COMPLETE")
print("="*80)