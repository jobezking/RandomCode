'''Step 1: Combine datasets with mandatory feature enforcement
We’ll union all columns, but mark DS2’s unique features as mandatory.
'''
# combine_with_mandatory.py
import pandas as pd

MANDATORY_DS2_FEATURES = ['garage_spaces']  # Example: DS2-only mandatory feature

def combine_datasets():
    ds1 = pd.read_csv("housing_ds1.csv")
    ds2 = pd.read_csv("housing_ds2.csv")
    ds3 = pd.read_csv("housing_ds3.csv")

    ds1['source'] = 'DS1'
    ds2['source'] = 'DS2'
    ds3['source'] = 'DS3'

    combined = pd.concat([ds1, ds2, ds3], axis=0, ignore_index=True)

    # Check mandatory features
    for feat in MANDATORY_DS2_FEATURES:
        if feat not in combined.columns:
            raise ValueError(f"Mandatory feature {feat} missing in combined dataset!")

    print("Combined shape:", combined.shape)
    return combined

if __name__ == "__main__":
    combined = combine_datasets()
    combined.to_csv("housing_combined.csv", index=False)
    print("Saved housing_combined.csv with mandatory DS2 features enforced")

'''
Step 2: Linear Regression on combined dataset
We’ll run a baseline regression, but require that mandatory DS2 features are present in training and prediction.
'''
# linear_regression_mandatory.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score, mean_squared_error

MANDATORY_DS2_FEATURES = ['garage_spaces']

df = pd.read_csv("housing_combined.csv")

# Feature engineering
df['age'] = 2025 - df['construction_year']
if 'renovation_year' in df.columns:
    df['reno_age'] = np.where(df['renovation_year'].notna(),
                              2025 - df['renovation_year'],
                              df['age'])
else:
    df['reno_age'] = df['age']

y = df['price']
X = df.drop(columns=['price'])

# Ensure mandatory features exist
for feat in MANDATORY_DS2_FEATURES:
    if feat not in X.columns:
        raise ValueError(f"Mandatory feature {feat} missing in dataset!")

# Preprocessing
numeric_cols = X.select_dtypes(include=['int64','float64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

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
pipe = Pipeline(steps=[
    ('pre', preprocessor),
    ('model', LinearRegression())
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipe.fit(X_train, y_train)
preds = pipe.predict(X_test)

print("Baseline Linear Regression")
print("R2:", r2_score(y_test, preds))
print("RMSE:", mean_squared_error(y_test, preds, squared=False))
'''Step 3: Tuning with Ridge (regularized linear regression)
'''
from sklearn.model_selection import GridSearchCV

ridge_pipe = Pipeline(steps=[
    ('pre', preprocessor),
    ('model', Ridge())
])

param_grid = {'model__alpha': [0.1, 1.0, 10.0, 50.0]}
ridge_search = GridSearchCV(ridge_pipe, param_grid, cv=5,
                            scoring='neg_root_mean_squared_error', n_jobs=-1)
ridge_search.fit(X_train, y_train)

print("\nBest Ridge Params:", ridge_search.best_params_)
print("Ridge R2:", r2_score(y_test, ridge_search.predict(X_test)))
print("Ridge RMSE:", mean_squared_error(y_test, ridge_search.predict(X_test), squared=False))
'''
Step 4: Driver with mandatory feature enforcement
'''
# driver_mandatory.py
import pandas as pd
from linear_regression_mandatory import pipe, MANDATORY_DS2_FEATURES

class HousingPredictorMandatory:
    def __init__(self, model):
        self.model = model
        self.mandatory = MANDATORY_DS2_FEATURES

    def predict(self, features: dict):
        # Enforce mandatory features
        for feat in self.mandatory:
            if feat not in features or features[feat] is None:
                raise ValueError(f"Missing mandatory feature: {feat}")

        X = pd.DataFrame([features])
        return float(self.model.predict(X)[0])

if __name__ == "__main__":
    predictor = HousingPredictorMandatory(pipe)

    # Example: must include garage_spaces
    sample_input = {
        'square_footage': 2400,
        'rooms': 4,
        'bathrooms': 2.5,
        'zip_code': '30328',
        'broadband': 1,
        'garage_spaces': 2  # mandatory DS2 feature
    }

    price = predictor.predict(sample_input)
    print(f"Predicted price: ${price:,.0f}")
'''
Key Takeaways
Mandatory DS2 features (e.g., garage_spaces) must always be provided by the user.
If missing, the driver raises an error instead of imputing.
The combined dataset still benefits from DS1 and DS3, but DS2’s unique features are enforced.
Ridge tuning stabilizes coefficients and improves generalization.
'''
#
'''
let’s walk through how you can visualize the coefficient magnitudes from your tuned Ridge Regression model, 
so you can see exactly how much weight the model assigns to each feature. 
This is especially useful for comparing DS2‑specific mandatory features like garage_spaces against universal ones like square_footage.
'''
#
'''Step 1: Extract coefficients and feature names
After fitting your Ridge model inside the pipeline, you can pull out the coefficients and align them with the transformed feature names:
'''
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Get the trained Ridge model from your GridSearchCV
best_ridge = ridge_search.best_estimator_

# Extract preprocessing and model
pre = best_ridge.named_steps['pre']
ridge_model = best_ridge.named_steps['model']

# Numeric and categorical columns
num_cols = pre.transformers_[0][2]
cat_cols = pre.transformers_[1][2]

# Get one-hot encoded feature names
ohe = pre.named_transformers_['cat'].named_steps['onehot']
cat_feature_names = ohe.get_feature_names_out(cat_cols)

# Combine into full feature name list
feature_names = list(num_cols) + list(cat_feature_names)

# Align coefficients
coefs = pd.Series(ridge_model.coef_, index=feature_names)

# Sort by absolute value
top_coefs = coefs.reindex(coefs.abs().sort_values(ascending=False).index)[:15]
'''Step 2: Plot the top coefficients'''
plt.figure(figsize=(8,6))
top_coefs.sort_values().plot(kind='barh', color='teal')
plt.title("Top 15 Ridge Regression Coefficients (by absolute value)")
plt.xlabel("Coefficient magnitude")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()
'''Step 3: Interpretation
Positive coefficients → increase predicted price when the feature increases (e.g., square_footage, bathrooms, garage_spaces).

Negative coefficients → decrease predicted price (e.g., age, hoa_fee).

DS2‑specific features like garage_spaces will appear in this chart if they carry predictive weight. If the bar is large, it means the model strongly relies on it.

Universal features like square_footage almost always dominate, but Ridge regularization ensures no single feature overwhelms the model.
This visualization helps you understand how much influence each feature has on the predicted housing prices, especially highlighting the importance of mandatory DS2 features in the context of the combined dataset.'''
# Step 1: Extract coefficients and feature names
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Get the trained Ridge model from your GridSearchCV
best_ridge = ridge_search.best_estimator_
# Extract preprocessing and model
pre = best_ridge.named_steps['pre']
ridge_model = best_ridge.named_steps['model']
# Numeric and categorical columns
num_cols = pre.transformers_[0][2]
cat_cols = pre.transformers_[1][2]
# Get one-hot encoded feature names
ohe = pre.named_transformers_['cat'].named_steps['onehot']
cat_feature_names = ohe.get_feature_names_out(cat_cols)
# Combine into full feature name list
feature_names = list(num_cols) + list(cat_feature_names)
# Align coefficients
coefs = pd.Series(ridge_model.coef_, index=feature_names)
# Sort by absolute value
top_coefs = coefs.reindex(coefs.abs().sort_values(ascending=False).index[:15])
# Step 2: Plot the top coefficients
plt.figure(figsize=(8,6))
top_coefs.sort_values().plot(kind='barh', color='teal')
plt.title("Top 15 Ridge Regression Coefficients (by absolute value)")
plt.xlabel("Coefficient magnitude")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()
# Step 3: Interpretation
# Positive coefficients → increase predicted price when the feature increases (e.g., square_footage
# Negative coefficients → decrease predicted price (e.g., age, hoa_fee)
# DS2‑specific features like garage_spaces will appear in this chart if they carry predictive weight
# Universal features like square_footage almost always dominate, but Ridge regularization ensures no single
# feature overwhelms the model