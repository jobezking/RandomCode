#Generating Sample Housing Datasets
import pandas as pd
import numpy as np
import os

# --- Dataset 1: The most complete dataset ---
np.random.seed(42)
n_samples = 500
sqft = np.random.randint(900, 3500, n_samples)
rooms = np.random.randint(2, 6, n_samples)
bathrooms = np.random.randint(1, 5, n_samples)
construction_date = np.random.randint(1950, 2022, n_samples)

# Ensure renovation_date is after construction_date
renovation_date = [d + np.random.randint(1, 15) if np.random.rand() > 0.6 else d for d in construction_date]

zip_code = np.random.choice([30308, 30309, 30327, 30305, 30342], n_samples)
floors = np.random.randint(1, 4, n_samples)
broadband_access = np.random.randint(0, 2, n_samples)

# Price formula with noise
price = (sqft * 150) + (rooms * 5000) + (bathrooms * 7500) + ((zip_code - 30300) * 1000) + \
        ((np.array(renovation_date) - 1950) * 500) + (broadband_access * 10000) + np.random.normal(0, 25000, n_samples)
price = price.astype(int)

df1 = pd.DataFrame({
    'sqft': sqft,
    'rooms': rooms,
    'bathrooms': bathrooms,
    'construction_date': construction_date,
    'renovation_date': renovation_date,
    'zip_code': zip_code,
    'floors': floors,
    'broadband_access': broadband_access,
    'price': price
})

# --- Dataset 2: Missing 'renovation_date' ---
df2 = df1.drop(columns=['renovation_date']).copy()
# Adjust price slightly as if renovation data was never a factor
df2['price'] = (df2['price'] * 0.95).astype(int)

# --- Dataset 3: Missing 'broadband_access' and adds 'has_pool' ---
df3 = df1.drop(columns=['broadband_access']).copy()
df3['has_pool'] = np.random.randint(0, 2, n_samples)
# Adjust price based on the new feature
df3['price'] = (df3['price'] + df3['has_pool'] * 20000).astype(int)

# --- Save to CSV files ---
if not os.path.exists('data'):
    os.makedirs('data')

df1.to_csv('data/atlanta_housing_v1.csv', index=False)
df2.to_csv('data/atlanta_housing_v2.csv', index=False)
df3.to_csv('data/atlanta_housing_v3.csv', index=False)

print("Generated 3 CSV files in 'data/' directory.")
print("Dataset 1 Head:\n", df1.head())

#Data Loading, Analysis, and Preprocessing
import matplotlib.pyplot as plt
import seaborn as sns

# Load the primary dataset
df = pd.read_csv('data/atlanta_housing_v1.csv')

print("--- Data Info ---")
df.info()

print("\n--- Descriptive Statistics ---")
print(df.describe())

#Exploratory Data Analysis (EDA)
#Price Distribution: Let's see how our target variable, price, is distributed.
plt.figure(figsize=(10, 6))
sns.histplot(df['price'], kde=True, bins=30)
plt.title('Distribution of House Prices')
plt.xlabel('Price ($)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

#Feature Correlation: A heatmap helps us see which features are most correlated with price.
plt.figure(figsize=(12, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Feature Correlation Matrix')
plt.show()

#Data Preprocessing and Feature Engineering
#1. Feature Engineering: Dates are not useful as raw numbers. We can engineer a more meaningful feature like house_age.
#2. Categorical Encoding: Machine learning models require numerical input. 
# We must convert the categorical zip_code feature into a numerical format using One-Hot Encoding.
from datetime import datetime

# 1. Feature Engineering
current_year = datetime.now().year
df['house_age'] = current_year - df['construction_date']
df['renovated_since_construction'] = (df['renovation_date'] > df['construction_date']).astype(int)

# Drop original date columns as they are now represented by age and renovation status
df = df.drop(columns=['construction_date', 'renovation_date'])

# 2. One-Hot Encoding for 'zip_code'
df = pd.get_dummies(df, columns=['zip_code'], prefix='zip')

print("\n--- Processed DataFrame Head ---")
print(df.head())
'''
3. Model Training 🤖
We will now train three different models:

Linear Regression: A straightforward model that finds a linear relationship between features and the target.

Random Forest Regressor: An ensemble model that builds multiple decision trees and merges them to get a more accurate and stable prediction.

Support Vector Regressor (SVR): A model that finds a hyperplane in an N-dimensional space that best fits the data.
'''
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, r2_score

# Define features (X) and target (y)
X = df.drop('price', axis=1)
y = df['price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
# Scaling is crucial for SVR and good practice for Linear Regression.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Initialize and Train Models ---
# Model 1: Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)

# Model 2: Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train) # Note: RF doesn't strictly need scaling but it's fine to use it

# Model 3: Support Vector Regressor (SVR)
svr_model = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr_model.fit(X_train_scaled, y_train)

# --- Evaluate Models ---
models = {
    "Linear Regression": lr_model,
    "Random Forest": rf_model,
    "SVR": svr_model
}

print("\n--- Model Performance ---")
for name, model in models.items():
    y_pred = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Model: {name}")
    print(f"  Mean Absolute Error (MAE): ${mae:,.2f}")
    print(f"  R-squared ($R^2$): {r2:.4f}\n")

'''
Analysis:

The Random Forest model likely performed the best, indicated by the highest R2 score and lowest Mean Absolute Error (MAE). 
This is because it can capture complex, non-linear relationships in the data that Linear Regression cannot.
Linear Regression provides a solid baseline.
SVR's performance is highly dependent on its parameters (C, gamma).
'''

'''
4. Simulating User Input and Making Predictions
Here is a driver function that simulates a user providing details for a new house listing. 
It preprocesses the input to match the training data format and then predicts the price using all three trained models.
'''
def predict_new_listing(listing_details, models_dict, feature_columns, data_scaler):
    """
    Predicts the price of a new housing listing.

    Args:
        listing_details (dict): A dictionary of house features.
        models_dict (dict): Dictionary containing the trained models.
        feature_columns (list): The list of column names from the training data (X.columns).
        data_scaler (StandardScaler): The fitted scaler from training.

    Returns:
        dict: A dictionary with predictions from each model.
    """
    # Create a DataFrame from the input
    input_df = pd.DataFrame([listing_details])

    # --- Preprocess input to match training data format ---
    # 1. Feature Engineering
    input_df['house_age'] = datetime.now().year - input_df['construction_date']
    input_df['renovated_since_construction'] = (input_df['renovation_date'] > input_df['construction_date']).astype(int)
    input_df = input_df.drop(columns=['construction_date', 'renovation_date'])
    
    # 2. One-Hot Encoding
    input_df = pd.get_dummies(input_df, columns=['zip_code'], prefix='zip')

    # 3. Align Columns: Ensure input_df has the same columns as the training data
    # This adds missing one-hot encoded columns and fills them with 0
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)

    # 4. Scale the data
    input_scaled = data_scaler.transform(input_df)

    # --- Make Predictions ---
    predictions = {}
    for name, model in models_dict.items():
        pred_price = model.predict(input_scaled)[0]
        predictions[name] = f"${pred_price:,.2f}"

    return predictions

# --- Example Usage ---
# Simulate user input for a house that isn't in our dataset
new_house = {
    'sqft': 2200,
    'rooms': 4,
    'bathrooms': 3,
    'construction_date': 1995,
    'renovation_date': 2018,
    'zip_code': 30309, # A zip code the model has seen
    'floors': 2,
    'broadband_access': 1,
}

predicted_prices = predict_new_listing(new_house, models, X.columns, scaler)

print("\n--- 🏡 Predicted Prices for New Listing ---")
for model_name, price in predicted_prices.items():
    print(f"{model_name}: {price}")

'''
5. Customizing a Model: A Random Forest Tutorial 🛠️
Let's pick the Random Forest Regressor and customize it to potentially improve its performance. Customization is done through hyperparameter tuning.

Hyperparameters are settings you can change to control the model's behavior before it starts training. For a Random Forest, key hyperparameters include:

n_estimators: The number of decision trees in the forest. More trees can improve performance but increase computation time.

max_depth: The maximum number of levels in each decision tree. Deeper trees can capture more complex patterns but risk overfitting 
(learning the training data too well and failing on new data).

min_samples_split: The minimum number of data points required to split a node in a tree. This helps control overfitting.

min_samples_leaf: The minimum number of data points allowed in a leaf node.

How to Find the Best Hyperparameters?
We'll use Grid Search with Cross-Validation (GridSearchCV).

Grid Search: You define a "grid" of hyperparameters you want to test (e.g., n_estimators of 100, 200, 300). 
It will systematically train and evaluate a model for every possible combination.

Cross-Validation (CV): To prevent our tuning from being biased towards our specific test set, 
CV splits the training data into 'k' smaller sets or "folds." It trains the model on k-1 folds and tests it on the remaining fold, 
rotating through all folds. This gives a more robust measure of performance.
'''
from sklearn.model_selection import GridSearchCV

# 1. Define the parameter grid
# This tells GridSearchCV which parameters and values to test.
param_grid = {
    'n_estimators': [100, 150, 200],      # Number of trees
    'max_depth': [10, 20, 30, None],        # Max depth of trees (None means no limit)
    'min_samples_split': [2, 5, 10],       # Min samples to split a node
    'min_samples_leaf': [1, 2, 4]          # Min samples at a leaf node
}

# 2. Instantiate GridSearchCV
# We create a new RF model instance to tune.
# cv=5 means 5-fold cross-validation.
# n_jobs=-1 uses all available CPU cores to speed up the process.
# scoring='neg_mean_absolute_error' is used because GridSearchCV tries to maximize a score.
# By maximizing the negative MAE, we are effectively minimizing the MAE.
grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=42),
                           param_grid=param_grid,
                           cv=5,
                           scoring='neg_mean_absolute_error',
                           n_jobs=-1,
                           verbose=2)

# 3. Fit the grid search to the data
# This will be computationally intensive as it's training many models!
print("\n--- Starting Hyperparameter Tuning with GridSearchCV ---")
grid_search.fit(X_train_scaled, y_train)

# 4. Get the best parameters and the best model
print("\n--- Tuning Complete ---")
print(f"Best Hyperparameters Found: {grid_search.best_params_}")

# The best model is automatically refit on the entire training data
best_rf_model = grid_search.best_estimator_

# 5. Evaluate the newly tuned model
y_pred_tuned = best_rf_model.predict(X_test_scaled)
mae_tuned = mean_absolute_error(y_test, y_pred_tuned)
r2_tuned = r2_score(y_test, y_pred_tuned)

print("\n--- Tuned Random Forest Performance ---")
print(f"  Tuned Mean Absolute Error (MAE): ${mae_tuned:,.2f}")
print(f"  Tuned R-squared ($R^2$): {r2_tuned:.4f}\n")

# 6. Predict with the new, optimized model
tuned_predictions = predict_new_listing(new_house, {"Tuned Random Forest": best_rf_model}, X.columns, scaler)
print("--- 🌟 Prediction from Tuned Model ---")
print(f"Tuned Random Forest: {tuned_predictions['Tuned Random Forest']}")