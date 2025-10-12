#Dataset generation and schema differences
# generate_datasets.py
import numpy as np
import pandas as pd

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

if __name__ == "__main__":
    ds1 = generate_ds1()
    ds2 = generate_ds2()
    ds3 = generate_ds3()
    ds1.to_csv("housing_ds1.csv", index=False)
    ds2.to_csv("housing_ds2.csv", index=False)
    ds3.to_csv("housing_ds3.csv", index=False)
    print("Saved: housing_ds1.csv, housing_ds2.csv, housing_ds3.csv")

#Exploratory data analysis and visualizations
# eda.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def quick_eda(df, title="Dataset"):
    print(f"\n=== {title} ===")
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))
    print(df.describe(include='all'))

    # Missing overview
    missing = df.isna().mean().sort_values(ascending=False)
    print("\nMissing rate (%):")
    print((missing * 100).round(2))

    # Price vs square footage
    plt.figure(figsize=(6,4))
    sns.scatterplot(data=df, x='square_footage', y='price', alpha=0.4)
    plt.title(f'{title}: Price vs Square Footage')
    plt.tight_layout()
    plt.show()

    # Price by bathrooms (binned)
    if 'bathrooms' in df.columns:
        plt.figure(figsize=(6,4))
        sns.boxplot(data=df, x='bathrooms', y='price')
        plt.title(f'{title}: Price by Bathrooms')
        plt.tight_layout()
        plt.show()

    # Price distribution by ZIP
    plt.figure(figsize=(8,4))
    sns.boxplot(data=df, x='zip_code', y='price')
    plt.title(f'{title}: Price by ZIP code')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    ds1 = pd.read_csv("housing_ds1.csv")
    ds2 = pd.read_csv("housing_ds2.csv")
    ds3 = pd.read_csv("housing_ds3.csv")
    quick_eda(ds1, "DS1")
    quick_eda(ds2, "DS2")
    quick_eda(ds3, "DS3")
'''
Modeling across three datasets
We’ll build a consistent preprocessing pipeline per dataset, then fit:
Linear Regression, Random Forest Regressor, Gradient Boosting Regressor (sklearn; not XGBoost)
Each model gets 3 results by training on DS1, DS2, and DS3 separately.
'''
# train_models.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

def load_datasets():
    ds1 = pd.read_csv("housing_ds1.csv")
    ds2 = pd.read_csv("housing_ds2.csv")
    ds3 = pd.read_csv("housing_ds3.csv")
    return {"DS1": ds1, "DS2": ds2, "DS3": ds3}

def make_features(df):
    df = df.copy()
    # Convert years to ages (robust to missing renovation_year)
    df['age'] = 2025 - df['construction_year']
    if 'renovation_year' in df.columns:
        df['reno_age'] = np.where(df['renovation_year'].notna(), 2025 - df['renovation_year'], df['age'])
    else:
        df['reno_age'] = df['age']
    return df

def split_features_target(df):
    y = df['price']
    X = df.drop(columns=['price'])
    return X, y

def build_preprocessor(X):
    # Identify column types dynamically
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

    # We know 'zip_code' is categorical; ensure included
    if 'zip_code' in X.columns and 'zip_code' not in categorical_cols:
        categorical_cols.append('zip_code')

    # Pipelines
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
    return preprocessor

def evaluate(model, X_test, y_test, name, ds_name):
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    rmse = mean_squared_error(y_test, preds, squared=False)
    print(f"{name} on {ds_name} -> R2: {r2:.3f}, RMSE: ${rmse:,.0f}")
    return r2, rmse

def train_all():
    datasets = load_datasets()
    results = []
    trained = {}

    for ds_name, df in datasets.items():
        df = make_features(df)
        X, y = split_features_target(df)

        preprocessor = build_preprocessor(X)

        # Models
        models = {
            'LinearRegression': Pipeline(steps=[
                ('pre', preprocessor),
                ('model', LinearRegression())
            ]),
            'RandomForest': Pipeline(steps=[
                ('pre', preprocessor),
                ('model', RandomForestRegressor(
                    n_estimators=300, max_depth=None, random_state=42, n_jobs=-1
                ))
            ]),
            'GradientBoosting': Pipeline(steps=[
                ('pre', preprocessor),
                ('model', GradientBoostingRegressor(
                    n_estimators=300, learning_rate=0.05, max_depth=3, random_state=42
                ))
            ])
        }

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        for name, pipe in models.items():
            pipe.fit(X_train, y_train)
            r2, rmse = evaluate(pipe, X_test, y_test, name, ds_name)
            results.append({'dataset': ds_name, 'model': name, 'R2': r2, 'RMSE': rmse})
            trained[(ds_name, name)] = pipe

    # Summarize
    print("\nSummary (higher R2, lower RMSE are better):")
    print(pd.DataFrame(results).sort_values(['model','dataset']))

    return trained, pd.DataFrame(results)

if __name__ == "__main__":
    train_all()
'''
Driver class: partial user input to predicted price
This class:
Loads the trained pipes returned by training.
Accepts a subset of features, imputes missing with training medians/modes via the pipeline.
Predicts with a selected dataset/model combo.
'''
# driver.py
import pandas as pd
from train_models import train_all

class HousingPricePredictor:
    def __init__(self):
        self.models, self.results = train_all()

    def available_models(self):
        return sorted(set([m for (_, m) in self.models.keys()]))

    def available_datasets(self):
        return sorted(set([d for (d, _) in self.models.keys()]))

    def predict(self, features: dict, dataset: str = "DS1", model: str = "GradientBoosting"):
        key = (dataset, model)
        if key not in self.models:
            raise ValueError(f"Model {model} on {dataset} not available. "
                             f"Use dataset in {self.available_datasets()} and model in {self.available_models()}.")

        pipe = self.models[key]

        # Build single-row DataFrame; missing cols left NaN for pipeline imputers
        # We rely on the pipeline's ColumnTransformer to find columns from training schema.
        # So we need to align with training X columns: reconstruct by inspecting preprocessor.
        pre = pipe.named_steps['pre']
        # Extract feature names from training data via transformers' column lists
        num_cols = pre.transformers_[0][2]
        cat_cols = pre.transformers_[1][2]
        all_cols = list(num_cols) + list(cat_cols)

        row = {col: features.get(col, None) for col in all_cols}
        X = pd.DataFrame([row])

        pred = pipe.predict(X)[0]
        return float(pred)

if __name__ == "__main__":
    predictor = HousingPricePredictor()
    print("Models:", predictor.available_models())
    print("Datasets:", predictor.available_datasets())

    # Example partial inputs (missing many features intentionally)
    sample_input = {
        'square_footage': 2600,
        'rooms': 4,
        'bathrooms': 2.5,
        'zip_code': '30328',
        'broadband': 1
        # Missing floors, renovation_year, construction_year, etc.
    }

    price = predictor.predict(sample_input, dataset="DS1", model="GradientBoosting")
    print(f"Predicted price (DS1, GradientBoosting): ${price:,.0f}")

    price2 = predictor.predict(sample_input, dataset="DS2", model="RandomForest")
    print(f"Predicted price (DS2, RandomForest): ${price2:,.0f}")

    price3 = predictor.predict(sample_input, dataset="DS3", model="LinearRegression")
    print(f"Predicted price (DS3, LinearRegression): ${price3:,.0f}")
