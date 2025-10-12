'''
Step 1: Combine the datasets
Since the three datasets have slightly different feature sets, we’ll align them by taking the union of all columns. 
Missing values will be handled with imputation.
'''
# combine_datasets.py
import pandas as pd

def combine_datasets():
    ds1 = pd.read_csv("housing_ds1.csv")
    ds2 = pd.read_csv("housing_ds2.csv")
    ds3 = pd.read_csv("housing_ds3.csv")

    # Add dataset identifier (optional, could be useful for analysis)
    ds1['source'] = 'DS1'
    ds2['source'] = 'DS2'
    ds3['source'] = 'DS3'

    # Union of all columns
    combined = pd.concat([ds1, ds2, ds3], axis=0, ignore_index=True)

    print("Combined shape:", combined.shape)
    print("Columns:", combined.columns.tolist())
    return combined

if __name__ == "__main__":
    combined = combine_datasets()
    combined.to_csv("housing_combined.csv", index=False)
    print("Saved housing_combined.csv")
'''
Step 2: Run baseline Linear Regression
We’ll preprocess the combined dataset and run a basic Linear Regression.
'''
# linear_regression_combined.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error

# Load combined dataset
df = pd.read_csv("housing_combined.csv")

# Feature engineering: age and reno_age
df['age'] = 2025 - df['construction_year']
if 'renovation_year' in df.columns:
    df['reno_age'] = np.where(df['renovation_year'].notna(),
                              2025 - df['renovation_year'],
                              df['age'])
else:
    df['reno_age'] = df['age']

# Split features/target
y = df['price']
X = df.drop(columns=['price'])

# Identify column types
numeric_cols = X.select_dtypes(include=['int64','float64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

# Preprocessor
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

# Baseline Linear Regression
baseline_pipe = Pipeline(steps=[
    ('pre', preprocessor),
    ('model', LinearRegression())
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
baseline_pipe.fit(X_train, y_train)
preds = baseline_pipe.predict(X_test)

print("Baseline Linear Regression")
print("R2:", r2_score(y_test, preds))
print("RMSE:", mean_squared_error(y_test, preds, squared=False))
'''
Step 3: Tune Linear Regression (Ridge & Lasso)
Plain Linear Regression has no hyperparameters to tune, but we can regularize it with Ridge (L2) and Lasso (L1). 
This helps handle multicollinearity and improves generalization.
'''
from sklearn.model_selection import GridSearchCV

# Ridge Regression
ridge_pipe = Pipeline(steps=[
    ('pre', preprocessor),
    ('model', Ridge())
])

ridge_params = {'model__alpha': [0.1, 1.0, 10.0, 50.0, 100.0]}
ridge_search = GridSearchCV(ridge_pipe, ridge_params, cv=5,
                            scoring='neg_root_mean_squared_error', n_jobs=-1)
ridge_search.fit(X_train, y_train)

print("\nBest Ridge Params:", ridge_search.best_params_)
print("Ridge R2:", r2_score(y_test, ridge_search.predict(X_test)))
print("Ridge RMSE:", mean_squared_error(y_test, ridge_search.predict(X_test), squared=False))

# Lasso Regression
lasso_pipe = Pipeline(steps=[
    ('pre', preprocessor),
    ('model', Lasso(max_iter=5000))
])

lasso_params = {'model__alpha': [0.001, 0.01, 0.1, 1.0, 10.0]}
lasso_search = GridSearchCV(lasso_pipe, lasso_params, cv=5,
                            scoring='neg_root_mean_squared_error', n_jobs=-1)
lasso_search.fit(X_train, y_train)

print("\nBest Lasso Params:", lasso_search.best_params_)
print("Lasso R2:", r2_score(y_test, lasso_search.predict(X_test)))
print("Lasso RMSE:", mean_squared_error(y_test, lasso_search.predict(X_test), squared=False))
''' 
Step 4: Inspect coefficients (ridge_search.best_estimator_.named_steps['model'].coef_) to see which features matter most.
'''
# Inspect Ridge coefficients
ridge_model = ridge_search.best_estimator_.named_steps['model']
feature_names = (numeric_cols +
                 list(ridge_search.best_estimator_.named_steps['pre']
                      .transformers_[1][1]
                      .named_steps['onehot']
                      .get_feature_names_out(categorical_cols)))
coefficients = pd.Series(ridge_model.coef_, index=feature_names).sort_values()
print("\nTop 10 Positive Coefficients:\n", coefficients.tail(10))
print("\nTop 10 Negative Coefficients:\n", coefficients.head(10))
'''
'''