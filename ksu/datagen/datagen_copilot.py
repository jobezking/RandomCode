#Dataset generation and schema differences
# generate_datasets.py
import numpy as np
import pandas as pd
import os

rng = np.random.default_rng(42)

def gen_zip_codes(n):
    # Atlanta-style ZIPs range example; categorical variety
    zips = rng.choice([30301, 30305, 30309, 30318, 30328, 30338, 30030, 30080, 30144], size=n)
    return zips.astype(str)

def gen_broadband(n, p=0.75):
    return rng.choice([0, 1], size=n, p=[1-p, p])

def base_core(n):
    sf = rng.normal(2200, 600, size=n).clip(600, 5000).round()
    rooms = rng.integers(3, 8, size=n)
    baths = rng.choice([1, 1.5, 2, 2.5, 3, 3.5], size=n, p=[0.05, 0.1, 0.25, 0.25, 0.25, 0.10])
    construction_year = rng.integers(1950, 2022, size=n)
    renovation_year = np.where(rng.random(n) < 0.4, rng.integers(1980, 2023, size=n), np.nan)
    zip_code = gen_zip_codes(n)
    floors = rng.integers(1, 3, size=n)
    broadband = gen_broadband(n, p=0.8)
    return pd.DataFrame({
        'square_footage': sf,
        'rooms': rooms,
        'bathrooms': baths,
        'construction_year': construction_year,
        'renovation_year': renovation_year,
        'zip_code': zip_code,
        'floors': floors,
        'broadband': broadband
    })

def price_function(df, extras=None, noise_scale=25000):
    # Feature engineering inside generator to define realistic pricing
    age = 2025 - df['construction_year']
    reno_age = np.where(df['renovation_year'].notna(), 2025 - df['renovation_year'], age)
    zip_factor = df['zip_code'].map({
        '30301': 1.08, '30305': 1.22, '30309': 1.25, '30318': 1.10,
        '30328': 1.18, '30338': 1.15, '30030': 1.20, '30080': 1.05, '30144': 0.95
    }).fillna(1.0)

    base = (
        85 * df['square_footage'] +
        15000 * df['rooms'] +
        22000 * df['bathrooms'] +
        -1200 * age +
        -800 * np.minimum(reno_age, age) + 
        8000 * df['floors'] +
        15000 * df['broadband'] +
        0
    )

    if extras and 'lot_size' in df:
        base += 6 * df['lot_size']
    if extras and 'energy_efficiency' in df:
        base += 8000 * df['energy_efficiency']
    if extras and 'garage_spaces' in df:
        base += 12000 * df['garage_spaces']
    if extras and 'hoa_fee' in df:
        base += -10 * df['hoa_fee']

    # Interaction for premium ZIPs
    base *= zip_factor

    noise = rng.normal(0, noise_scale, size=len(df))
    price = (base + noise).clip(50000, 2_500_000)
    return price.round(0)

def generate_ds1(n=1200):
    df = base_core(n)
    df['lot_size'] = rng.normal(12000, 4000, size=n).clip(2000, 30000).round()
    df['energy_efficiency'] = rng.integers(1, 6, size=n)  # 1-5 scale
    df['price'] = price_function(df, extras=True, noise_scale=30000)
    return df

def generate_ds2(n=1000):
    df = base_core(n)
    df = df.drop(columns=['renovation_year'])  # intentionally missing
    df['garage_spaces'] = rng.integers(0, 3, size=n)
    df['broadband'] = gen_broadband(n, p=0.65)  # different distribution
    df['price'] = price_function(df, extras=True, noise_scale=35000)
    return df

def generate_ds3(n=900):
    df = base_core(n)
    df['hoa_fee'] = rng.normal(75, 30, size=n).clip(0, 250).round(2)
    df['broadband'] = gen_broadband(n, p=0.9)
    df['price'] = price_function(df, extras=True, noise_scale=28000)
    return df

def save_datasets():
    if not os.path.exists('data'):
        os.makedirs('data')
    df1 = generate_ds1()
    df2 = generate_ds2()
    df3 = generate_ds3()
    df1.to_csv('data/housing_dataset_v1.csv', index=False)
    df2.to_csv('data/housing_dataset_v2.csv', index=False)
    df3.to_csv('data/housing_dataset_v3.csv', index=False)
    print("Datasets saved to 'data/' directory.")
    print("Dataset 1 Head:\n", df1.head())

if __name__ == "__main__":
    save_datasets()