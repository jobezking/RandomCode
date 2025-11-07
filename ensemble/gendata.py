import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Dataset 1: Basic features (50 rows, 4 columns + price)
print("Generating Dataset 1: Basic Housing Features...")
n_samples_1 = 50

data1 = {
    'square_feet': np.random.randint(800, 3500, n_samples_1),
    'bedrooms': np.random.randint(1, 6, n_samples_1),
    'bathrooms': np.random.uniform(1, 4, n_samples_1).round(1),
    'age_years': np.random.randint(0, 50, n_samples_1)
}

# Generate realistic prices based on features
base_price_1 = (data1['square_feet'] * 150 + 
                data1['bedrooms'] * 20000 + 
                data1['bathrooms'] * 15000 - 
                data1['age_years'] * 1000)
noise_1 = np.random.normal(0, 30000, n_samples_1)
data1['housing_price'] = (base_price_1 + noise_1).round(-3)  # Round to nearest thousand

df1 = pd.DataFrame(data1)
df1.to_csv('housing_prices_basic.csv', index=False)
print(f"✓ Created 'housing_prices_basic.csv' with {len(df1)} rows and {len(df1.columns)} columns")
print(f"  Features: {', '.join(df1.columns[:-1])}\n")

# Dataset 2: Extended features (75 rows, 6 columns + price)
print("Generating Dataset 2: Extended Housing Features...")
n_samples_2 = 75

data2 = {
    'square_feet': np.random.randint(900, 4000, n_samples_2),
    'bedrooms': np.random.randint(2, 7, n_samples_2),
    'bathrooms': np.random.uniform(1.5, 5, n_samples_2).round(1),
    'garage_spaces': np.random.randint(0, 4, n_samples_2),
    'lot_size_sqft': np.random.randint(2000, 15000, n_samples_2),
    'distance_to_city_miles': np.random.uniform(0.5, 30, n_samples_2).round(1)
}

# Generate realistic prices with more complex relationships
base_price_2 = (data2['square_feet'] * 160 + 
                data2['bedrooms'] * 18000 + 
                data2['bathrooms'] * 17000 + 
                data2['garage_spaces'] * 12000 +
                data2['lot_size_sqft'] * 5 -
                data2['distance_to_city_miles'] * 3000)
noise_2 = np.random.normal(0, 40000, n_samples_2)
data2['housing_price'] = (base_price_2 + noise_2).round(-3)

df2 = pd.DataFrame(data2)
df2.to_csv('housing_prices_extended.csv', index=False)
print(f"✓ Created 'housing_prices_extended.csv' with {len(df2)} rows and {len(df2.columns)} columns")
print(f"  Features: {', '.join(df2.columns[:-1])}\n")

# Dataset 3: Alternative features (100 rows, 5 columns + price)
print("Generating Dataset 3: Alternative Housing Features...")
n_samples_3 = 100

# Different feature set focusing on quality and amenities
data3 = {
    'square_feet': np.random.randint(1000, 3800, n_samples_3),
    'year_built': np.random.randint(1970, 2024, n_samples_3),
    'num_rooms': np.random.randint(4, 12, n_samples_3),
    'has_pool': np.random.choice([0, 1], n_samples_3, p=[0.7, 0.3]),
    'school_rating': np.random.randint(3, 11, n_samples_3)
}

# Generate realistic prices
current_year = 2024
age = current_year - data3['year_built']
base_price_3 = (data3['square_feet'] * 155 + 
                data3['num_rooms'] * 12000 + 
                data3['has_pool'] * 50000 +
                data3['school_rating'] * 15000 -
                age * 800)
noise_3 = np.random.normal(0, 35000, n_samples_3)
data3['housing_price'] = (base_price_3 + noise_3).round(-3)

df3 = pd.DataFrame(data3)
df3.to_csv('housing_prices_alternative.csv', index=False)
print(f"✓ Created 'housing_prices_alternative.csv' with {len(df3)} rows and {len(df3.columns)} columns")
print(f"  Features: {', '.join(df3.columns[:-1])}\n")

# Summary
print("=" * 60)
print("SUMMARY OF GENERATED DATASETS")
print("=" * 60)
print(f"\nDataset 1 (Basic):")
print(f"  Shape: {df1.shape}")
print(f"  Price range: ${df1['housing_price'].min():,.0f} - ${df1['housing_price'].max():,.0f}")

print(f"\nDataset 2 (Extended):")
print(f"  Shape: {df2.shape}")
print(f"  Price range: ${df2['housing_price'].min():,.0f} - ${df2['housing_price'].max():,.0f}")

print(f"\nDataset 3 (Alternative):")
print(f"  Shape: {df3.shape}")
print(f"  Price range: ${df3['housing_price'].min():,.0f} - ${df3['housing_price'].max():,.0f}")

print("\n✓ All datasets successfully generated!")
print("  Each dataset has 'housing_price' as the target variable.")
print("  All datasets are ready for linear regression modeling.")