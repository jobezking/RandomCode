import pandas as pd
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# --- 1. Generate Simulated Data ---
print("Generating simulated housing data...")

X, y = make_regression(
    n_samples=1000,
    n_features=10,
    n_informative=8,
    noise=25,
    random_state=42
)

# Convert to DataFrame for clarity
feature_names = [f"f{i}" for i in range(X.shape[1])]
df = pd.DataFrame(X, columns=feature_names)
target = pd.Series(y, name="price")

# Simulate 3 different data sources
X1 = df[["f0", "f1", "f2", "f3"]]              # House Specs
X2 = df[["f4", "f5", "f6"]]                    # Location Data
X3 = df[["f7", "f8", "f9", "f0"]]              # Economic Data (note overlap)

# --- 2. Train/Test Split ---
(X1_train, X1_test,
 X2_train, X2_test,
 X3_train, X3_test,
 y_train, y_test) = train_test_split(
    X1, X2, X3, target, test_size=0.2, random_state=42
)

# --- 3. Train Base Models ---
print("Training base models...")
model_1 = LinearRegression().fit(X1_train, y_train)
model_2 = LinearRegression().fit(X2_train, y_train)
model_3 = LinearRegression().fit(X3_train, y_train)

# --- 4. Generate Meta-Features (Stacking) ---
print("Generating out-of-fold meta-features...")

preds_1_train = cross_val_predict(LinearRegression(), X1_train, y_train, cv=5)
preds_2_train = cross_val_predict(LinearRegression(), X2_train, y_train, cv=5)
preds_3_train = cross_val_predict(LinearRegression(), X3_train, y_train, cv=5)

X_meta_train = pd.DataFrame({
    "m1": preds_1_train,
    "m2": preds_2_train,
    "m3": preds_3_train
})

# --- 5. Train Meta-Model ---
print("Training meta-model...")
meta_model = LinearRegression().fit(X_meta_train, y_train)

weights = meta_model.coef_
print(f"Meta-Model Weights: M1={weights[0]:.2f}, M2={weights[1]:.2f}, M3={weights[2]:.2f}")

# --- 6. Evaluate on Test Set ---
print("\n--- Evaluating Performance ---")

preds_1_test = model_1.predict(X1_test)
preds_2_test = model_2.predict(X2_test)
preds_3_test = model_3.predict(X3_test)

X_meta_test = pd.DataFrame({
    "m1": preds_1_test,
    "m2": preds_2_test,
    "m3": preds_3_test
})

final_predictions = meta_model.predict(X_meta_test)

mse_1 = mean_squared_error(y_test, preds_1_test)
mse_2 = mean_squared_error(y_test, preds_2_test)
mse_3 = mean_squared_error(y_test, preds_3_test)
mse_meta = mean_squared_error(y_test, final_predictions)

print(f"Model 1 (Specs):    {mse_1:,.2f}")
print(f"Model 2 (Location): {mse_2:,.2f}")
print(f"Model 3 (Economic): {mse_3:,.2f}")
print("---------------------------------")
print(f"Ensemble Meta-Model: {mse_meta:,.2f}")

# --- 7. Inference Example ---
print("\n--- 🚀 New Prediction Example ---")

new_raw_data = df.iloc[X1_test.index[0]].values  # first test house
true_price = y_test.iloc[0]

# Split into same feature sets
new_data_1 = pd.DataFrame([new_raw_data[0:4]], columns=["f0","f1","f2","f3"])
new_data_2 = pd.DataFrame([new_raw_data[4:7]], columns=["f4","f5","f6"])
new_data_3 = pd.DataFrame([new_raw_data[[7,8,9,0]]], columns=["f7","f8","f9","f0"])

# Base predictions
pred_1 = model_1.predict(new_data_1)[0]
pred_2 = model_2.predict(new_data_2)[0]
pred_3 = model_3.predict(new_data_3)[0]

print(f"Base Predictions: M1={pred_1:.2f}, M2={pred_2:.2f}, M3={pred_3:.2f}")

# Meta-features
new_meta = pd.DataFrame([[pred_1, pred_2, pred_3]], columns=["m1","m2","m3"])
final_pred = meta_model.predict(new_meta)[0]

print("\nFinal Result")
print("---------------------------------")
print(f"Final Ensemble Prediction: ${final_pred:,.2f}")
print(f"Actual House Price:        ${true_price:,.2f}")
print("---------------------------------")
