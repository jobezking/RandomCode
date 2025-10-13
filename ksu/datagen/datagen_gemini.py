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
print("Dataset 2 Head:\n", df2.head())
print("Dataset 3 Head:\n", df3.head())