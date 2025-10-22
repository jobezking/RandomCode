#!/usr/bin/env python3

"""
DATA PREPROCESSING & FEATURE ENGINEERING CHECKLIST (V2)

This script provides an exhaustive checklist and example commands for processing
a CSV file loaded into a pandas DataFrame.

It covers the typical data science workflow from loading and visualization
to cleaning, feature engineering, and scaling.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    OneHotEncoder,
    OrdinalEncoder,
    PowerTransformer
)
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression


def create_sample_dataframe():
    """
    Creates a sample 'messy' DataFrame with common data problems
    like missing values, outliers, mixed types, and duplicates.
    """
    data = {
        'student_id': ['S001', 'S002', 'S003', 'S004', 'S005', 'S006', 'S007', 'S008', 'S009', 'S010', 'S001'],
        'age': [18, 19, 20, np.nan, 22, 21, 19, 20, 18, 999, 18],  # Missing, Outlier
        'gender': ['Male', 'Female', 'male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Male', 'Male'],  # Inconsistent
        'major': ['Physics', 'Art', 'Physics', 'Chemistry', 'Art', 'Physics', 'Chemistry', 'Art', 'Physics', 'Art', 'Physics'],
        'gpa': [3.8, 3.2, 3.9, 3.5, 3.0, 3.95, 3.6, 3.1, 3.85, 2.5, 3.8],
        'exam_score': [85, 78, 92, 88, 75, 95, 90, 77, 91, 65, 85],
        'satisfaction': ['High', 'Medium', 'High', 'Low', 'Medium', 'High', 'Low', 'Medium', 'High', 'Low', 'High'],  # Ordinal
        'study_hours_week': [10, 5, 12, 8, 3, 15, 9, 4, 11, 2, 10],
        'tuition_paid': [5000, 5000, 5200, 4800, 5100, 5000, 4900, 5000, 5200, 4800, 5000],  # Skewed
        'useless_col_1': [None] * 11,  # All missing
        'useless_col_2': ['Const'] * 11  # No variance
    }
    df = pd.DataFrame(data)
    return df


def main():
    """
    Main function to run the data processing checklist.
    """
    # --- CONFIGURATION ---
    # Set to True to display plots. You must close each plot window
    # to allow the script to continue.
    SHOW_PLOTS = False
    
    # ----------------------------------------------------------------------
    # --- 0. LOAD DATA & INITIAL INSPECTION ---
    # ----------------------------------------------------------------------
    # Checklist:
    # 1. Load the CSV file into a pandas DataFrame.
    # 2. Make a copy of the original data for processing.
    # 3. Perform an initial inspection.

    # In a real scenario, you would use:
    # df = pd.read_csv('your_file.name.csv')
    
    # For this example, we use our sample messy DataFrame
    df = create_sample_dataframe()
    df_processed = df.copy()

    print("--- 0. ORIGINAL DATAFRAME ---")
    print(df_processed.head())
    print("\n--- 0. ORIGINAL DATAFRAME INFO ---")
    df_processed.info()
    print("\n" + "="*80 + "\n")

    # ----------------------------------------------------------------------
    # --- 1. EXPLORATORY DATA ANALYSIS (EDA) & VISUALIZATION ---
    # ----------------------------------------------------------------------
    print("--- 1. EXPLORATORY DATA ANALYSIS (EDA) & VISUALIZATION ---")
    
    # Checklist:
    # 1. Get descriptive statistics for numerical data.
    # 2. Check value counts for categorical data.
    # 3. Visualize distributions of numerical features (Histograms, KDE plots).
    # 4. Visualize distributions of categorical features (Count plots).
    # 5. Visualize missing data patterns (Heatmap).
    # 6. Visualize relationships between variables (Scatter plots, Pair plots).
    # 7. Visualize correlations (Correlation heatmap).
    # 8. Visualize potential outliers (Box plots).

    # 1. Descriptive Statistics
    print("\nDescriptive Statistics:")
    print(df_processed.describe())
    
    # 2. Value Counts
    print(f"\nValue counts for 'gender':\n{df_processed['gender'].value_counts()}")
    
    if SHOW_PLOTS:
        print("Displaying plots... Close each plot window to continue.")
        
        # 3. Numerical Distributions (e.g., 'gpa' and 'age')
        sns.histplot(df_processed['gpa'], kde=True)
        plt.title("Distribution of GPA")
        plt.show()

        sns.histplot(df_processed['age'], kde=False)
        plt.title("Distribution of Age (Shows outlier 999)")
        plt.show()
        
        # 4. Categorical Distributions (e.g., 'major')
        sns.countplot(x='major', data=df_processed)
        plt.title("Count of Students by Major")
        plt.show()

        # 5. Missing Data Heatmap
        sns.heatmap(df_processed.isnull(), cbar=False, cmap='viridis', yticklabels=False)
        plt.title("Missing Value Heatmap")
        plt.show()
        
        # 7. Correlation Heatmap
        # (Note: .corr() only works on numeric data)
        numeric_cols = df_processed.select_dtypes(include=np.number)
        plt.figure(figsize=(10, 6))
        sns.heatmap(numeric_cols.corr(), annot=True, cmap='coolwarm', fmt='.2f')
        plt.title("Correlation Heatmap (Numeric Features)")
        plt.show()

        # 8. Outlier Box Plots
        sns.boxplot(x=df_processed['age'])
        plt.title("Boxplot of Age (Shows outlier 999)")
        plt.show()
        
        sns.boxplot(x=df_processed['exam_score'])
        plt.title("Boxplot of Exam Score")
        plt.show()
        
    print("EDA section complete.")
    print("\n" + "="*80 + "\n")

    # ----------------------------------------------------------------------
    # --- 2. DATA CLEANING (GENERAL) ---
    # ----------------------------------------------------------------------
    print("--- 2. DATA CLEANING (GENERAL) ---")
    
    # Checklist:
    # 1. Check for and handle duplicate rows.
    # 2. Check for and handle irrelevant columns (e.g., all NaN, no variance, or IDs).
    # 3. Check and correct data types (e.g., numbers stored as strings).
    # 4. Find and fix inconsistencies in categorical data (e.g., "Male" vs "male").
    # 5. Rename columns for clarity/consistency if needed.

    # 1. Handle Duplicates
    num_duplicates = df_processed.duplicated().sum()
    print(f"Found {num_duplicates} duplicate rows.")
    df_processed = df_processed.drop_duplicates(keep='first')

    # 2. Handle Irrelevant Columns
    df_processed = df_processed.dropna(axis=1, how='all')
    print("Dropped 'useless_col_1' (all NaN).")
    
    for col in df_processed.columns:
        if df_processed[col].nunique() == 1:
            df_processed = df_processed.drop(columns=[col])
            print(f"Dropped '{col}' (no variance).")
            
    if 'student_id' in df_processed.columns:
        df_processed = df_processed.drop(columns=['student_id'])
        print("Dropped 'student_id' (identifier).")

    # 4. Fix Inconsistencies
    if 'gender' in df_processed.columns:
        print(f"Unique 'gender' values before: {df_processed['gender'].unique()}")
        df_processed['gender'] = df_processed['gender'].str.lower().str.strip()
        print(f"Unique 'gender' values after: {df_processed['gender'].unique()}")

    # 5. Rename Columns
    df_processed = df_processed.rename(columns={'study_hours_week': 'study_hours'})
    print("Renamed 'study_hours_week' to 'study_hours'.")
    print("\n" + "="*80 + "\n")

    # ----------------------------------------------------------------------
    # --- 3. HANDLING MISSING VALUES ---
    # ----------------------------------------------------------------------
    print("--- 3. HANDLING MISSING VALUES ---")

    # Checklist:
    # 1. Identify missing values (NaNs).
    # 2. Decide on a strategy: drop or impute.
    # 3. Apply strategy (e.g., SimpleImputer for mean/median/mode, KNNImputer, ffill, dropna).

    # 1. Identify
    print("Missing values per column (after cleaning):")
    missing_data_summary = df_processed.isnull().sum()
    print(missing_data_summary[missing_data_summary > 0])

    # 3. Apply Strategy (Imputation for 'age')
    # We will use median for 'age' because it's robust to the '999' outlier
    median_imputer = SimpleImputer(strategy='median')
    df_processed['age'] = median_imputer.fit_transform(df_processed[['age']])
    print(f"Imputed missing 'age' values with median.")
    print("\n" + "="*80 + "\n")

    # ----------------------------------------------------------------------
    # --- 4. HANDLING OUTLIERS (STATISTICAL) ---
    # ----------------------------------------------------------------------
    print("--- 4. HANDLING OUTLIERS (STATISTICAL) ---")
    
    # Checklist:
    # 1. Use statistical methods to define outliers (Z-Score, IQR).
    # 2. Decide on a strategy: drop, cap (winsorize), or transform (see section 7).
    # 3. Apply the chosen strategy.

    # 1. Use IQR (Interquartile Range) for 'age' (which has the 999 value)
    Q1 = df_processed['age'].quantile(0.25)
    Q3 = df_processed['age'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    print(f"'age' IQR: {IQR}, Lower: {lower_bound}, Upper: {upper_bound}")
    outliers = df_processed[(df_processed['age'] < lower_bound) | (df_processed['age'] > upper_bound)]
    print(f"Found {len(outliers)} outliers in 'age' (e.g., 999) using IQR.")

    # 2. & 3. Handle (Capping/Winsorizing)
    # We'll cap 'age' at the upper bound.
    df_processed['age'] = df_processed['age'].clip(lower=lower_bound, upper=upper_bound)
    print(f"Capped 'age' outliers at {upper_bound}.")
    
    if SHOW_PLOTS:
        sns.boxplot(x=df_processed['age'])
        plt.title("Boxplot of Age (After Outlier Capping)")
        plt.show()
        
    print("\n" + "="*80 + "\n")

    # ----------------------------------------------------------------------
    # --- 5. HANDLING OUTLIERS (ADVANCED / MODEL-BASED) ---
    # ----------------------------------------------------------------------
    print("--- 5. HANDLING OUTLIERS (ADVANCED/MODEL-BASED) ---")

    # Checklist:
    # 1. Use model-based methods for multivariate outlier detection (Isolation Forest, LOF).
    # 2. Use a model's residuals (e.g., from Linear Regression) to find observations
    #    that don't fit the expected relationship.

    # 1. Using Isolation Forest
    # We'll check 'gpa' and 'exam_score' together
    features_for_iso = ['gpa', 'exam_score']
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    outlier_preds = iso_forest.fit_predict(df_processed[features_for_iso])
    
    # -1 means outlier, 1 means inlier
    df_processed['is_outlier_iso'] = outlier_preds
    print(f"Isolation Forest marked { (outlier_preds == -1).sum() } rows as outliers.")
    # In a real workflow, you might drop or investigate these:
    # df_processed = df_processed[df_processed['is_outlier_iso'] == 1]
    df_processed = df_processed.drop(columns=['is_outlier_iso']) # Dropping the flag for now
    
    # 2. Using Residuals
    # (e.g., 'exam_score' predicted by 'study_hours')
    X_res = df_processed[['study_hours']]
    y_res = df_processed['exam_score']
    model = LinearRegression()
    model.fit(X_res, y_res)
    residuals = y_res - model.predict(X_res)
    residual_std = residuals.std()
    
    # Find outliers: e.g., residuals > 3 standard deviations
    outliers_resid = df_processed[np.abs(residuals) > 3 * residual_std]
    print(f"Found {len(outliers_resid)} outliers based on regression residuals.")
    print("\n" + "="*80 + "\n")

    # ----------------------------------------------------------------------
    # --- 6. FEATURE ENGINEERING ---
    # ----------------------------------------------------------------------
    print("--- 6. FEATURE ENGINEERING ---")

    # Checklist:
    # 1. Create interaction features (e.g., col1 * col2).
    # 2. Create polynomial features (e.g., col1^2).
    # 3. Create new features based on domain knowledge (e.g., 'gpa_per_study_hour').
    # 4. Bin continuous variables into categories (e.g., 'age' into 'age_group').
    # 5. Create aggregate features (e.g., mean 'gpa' per 'major').

    # 3. Domain Knowledge Feature
    df_processed['gpa_per_study_hour'] = df_processed['gpa'] / (df_processed['study_hours'] + 1e-6)
    print("Created 'gpa_per_study_hour'.")

    # 4. Binning / Discretization
    df_processed['age_group'] = pd.cut(
        df_processed['age'],
        bins=[0, 18, 20, 22, np.inf],
        labels=['<=18', '19-20', '21-22', '23+']
    )
    print("Created 'age_group' by binning 'age'.")
    
    # 5. Aggregate Features
    avg_gpa_major = df_processed.groupby('major')['gpa'].transform('mean')
    df_processed['avg_gpa_by_major'] = avg_gpa_major
    print("Created 'avg_gpa_by_major'.")
    print("\n" + "="*80 + "\n")

    # ----------------------------------------------------------------------
    # --- 7. VARIABLE TRANSFORMATION ---
    # ----------------------------------------------------------------------
    print("--- 7. VARIABLE TRANSFORMATION ---")

    # Checklist:
    # 1. Identify skewed numerical variables (visualize with histplot/probplot).
    # 2. Apply transformations to normalize distribution (Log, Sqrt, Box-Cox, Yeo-Johnson).
    # 3. This helps algorithms that assume normality (like Linear Regression).

    # 1. Identify Skewness
    print(f"'tuition_paid' skewness: {df_processed['tuition_paid'].skew():.2f}")
    if SHOW_PLOTS:
        sns.histplot(df_processed['tuition_paid'], kde=True)
        plt.title("Original 'tuition_paid' Distribution")
        plt.show()
    
    # 2. Apply Transformation (Yeo-Johnson handles 0s and negatives)
    pt = PowerTransformer(method='yeo-johnson')
    df_processed['tuition_transformed'] = pt.fit_transform(df_processed[['tuition_paid']])
    print(f"'tuition_transformed' new skewness: {df_processed['tuition_transformed'].skew():.2f}")
    
    if SHOW_PLOTS:
        sns.histplot(df_processed['tuition_transformed'], kde=True)
        plt.title("Transformed 'tuition_paid' Distribution (Yeo-Johnson)")
        plt.show()
        
    print("Applied Yeo-Johnson transformation to 'tuition_paid'.")
    print("\n" + "="*80 + "\n")

    # ----------------------------------------------------------------------
    # --- 8. FEATURE CODING (ENCODING CATEGORICAL) ---
    # ----------------------------------------------------------------------
    print("--- 8. FEATURE CODING (ENCODING) ---")
    
    # Checklist:
    # 1. Identify ordinal variables (ordered) -> Manual Mapping or OrdinalEncoder.
    # 2. Identify nominal variables (unordered) -> One-Hot Encoding (pd.get_dummies).

    # 1. Ordinal Encoding ('satisfaction')
    satisfaction_map = {'Low': 1, 'Medium': 2, 'High': 3}
    df_processed['satisfaction_encoded'] = df_processed['satisfaction'].map(satisfaction_map)
    print("Applied manual ordinal encoding to 'satisfaction'.")
    df_processed = df_processed.drop(columns=['satisfaction']) # Drop original

    # 2. Nominal Encoding ('gender', 'major', 'age_group')
    nominal_cols = ['gender', 'major', 'age_group']
    df_processed = pd.get_dummies(
        df_processed,
        columns=nominal_cols,
        drop_first=True  # Avoids the dummy variable trap
    )
    print("Applied One-Hot Encoding to nominal columns.")
    print("\nTop 10 rows of processed DataFrame (post-encoding):")
    print(df_processed.head(10))
    print("\n" + "="*80 + "\n")
    
    # ----------------------------------------------------------------------
    # --- 9. FEATURE SCALING ---
    # ----------------------------------------------------------------------
    print("--- 9. FEATURE SCALING ---")

    # Checklist:
    # 1. IMPORTANT: Split data into training and testing sets FIRST.
    # 2. Identify numerical features to scale.
    # 3. Choose a scaler (StandardScaler, MinMaxScaler, RobustScaler).
    # 4. FIT the scaler on the TRAINING data only.
    # 5. TRANSFORM both the training and testing data.

    # 1. Split Data
    # Let's pretend 'gpa' is our target variable
    if 'gpa' in df_processed.columns:
        X = df_processed.drop(columns=['gpa'])
        y = df_processed['gpa']
        
        # Ensure all X columns are numeric before splitting
        X = X.select_dtypes(include=np.number)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        print(f"Split data into {X_train.shape[0]} train and {X_test.shape[0]} test samples.")

        # 2. Identify features to scale
        features_to_scale = [
            'age', 'exam_score', 'study_hours', 'tuition_paid',
            'gpa_per_study_hour', 'avg_gpa_by_major', 'tuition_transformed'
        ]
        # Filter list to only include columns that exist in X_train
        features_to_scale = [col for col in features_to_scale if col in X_train.columns]

        # 3. & 4. Choose and Fit Scaler
        # StandardScaler is a good default.
        # RobustScaler is good if you suspect outliers remain.
        scaler = StandardScaler()
        scaler.fit(X_train[features_to_scale])
        print("Fitted StandardScaler on training data.")

        # 5. Transform both sets
        X_train_scaled_arr = scaler.transform(X_train[features_to_scale])
        X_test_scaled_arr = scaler.transform(X_test[features_to_scale])

        # Convert back to DataFrame for clarity
        X_train_scaled = pd.DataFrame(X_train_scaled_arr, columns=features_to_scale, index=X_train.index)
        X_test_scaled = pd.DataFrame(X_test_scaled_arr, columns=features_to_scale, index=X_test.index)
        
        # Now, combine scaled numeric with unscaled (e.g., dummy) variables
        X_train_final = pd.concat(
            [X_train_scaled, X_train.drop(columns=features_to_scale)], axis=1
        )
        X_test_final = pd.concat(
            [X_test_scaled, X_test.drop(columns=features_to_scale)], axis=1
        )


        print("\n--- Scaled Training Data (Head) ---")
        print(X_train_final.head())
        print("\n--- Scaled Training Data (Describe) ---")
        print(X_train_final.describe())
    else:
        print("Target 'gpa' not found, skipping scaling example.")

    print("\n" + "="*80 + "\n")
    print("--- CHECKLIST COMPLETE ---")
    
    # Final step: Save the processed data
    # X_train_final.to_csv('X_train_final.csv', index=False)
    # y_train.to_csv('y_train.csv', index=False)
    print("Processing finished. (Data not saved in this example).")


if __name__ == "__main__":
    main()