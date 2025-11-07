import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import StackingRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# --- Global variables to hold the trained model and feature names ---
STACKING_MODEL = None
ALL_FEATURES = None

# Define the feature sets for each base model
FEATURES_BASIC = ['square_feet', 'bedrooms', 'bathrooms', 'age_years']
FEATURES_ALT = ['square_feet', 'year_built', 'num_rooms', 'has_pool', 'school_rating']
FEATURES_EXT = ['square_feet', 'bedrooms', 'bathrooms', 'garage_spaces', 'lot_size_sqft', 'distance_to_city_miles']

# List of CSV files to be combined
DATA_FILES = ['housing_prices_basic.csv', 'housing_prices_alternative.csv', 'housing_prices_extended.csv']

def train_model():
    """
    Loads data, builds, and trains the stacking ensemble model.
    """
    global STACKING_MODEL, ALL_FEATURES
    
    print("Loading and combining data files...")
    # 1. Load and combine all datasets
    df_list = []
    for f in DATA_FILES:
        try:
            df_list.append(pd.read_csv(f))
        except FileNotFoundError:
            print(f"Warning: File not found: {f}. Skipping.")
        except Exception as e:
            print(f"Error loading {f}: {e}. Skipping.")
            
    if not df_list:
        print("Error: No data files were successfully loaded. Exiting.")
        return

    full_df = pd.concat(df_list, ignore_index=True, sort=False)
    
    # 2. Prepare data for modeling
    # Drop rows where the target (housing_price) is missing
    full_df = full_df.dropna(subset=['housing_price'])
    
    X = full_df.drop('housing_price', axis=1)
    y = full_df['housing_price']
    
    # Store all unique feature names. This is crucial for the prediction function.
    ALL_FEATURES = X.columns.tolist()
    
    # Split data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Building pipelines...")
    # 3. Create the 3 base model pipelines (as requested in 2a & 2b)
    
    # Pipeline 1: Basic Features
    # This pipeline first selects only the 'FEATURES_BASIC' columns,
    # then imputes missing values (e.g., if 'bedrooms' is missing),
    # and finally fits a linear regression model.
    basic_transformer = ColumnTransformer(
        transformers=[
            ('imputer', SimpleImputer(strategy='mean'), FEATURES_BASIC)
        ],
        remainder='drop' # Drop columns not in FEATURES_BASIC
    )
    pipeline_basic = Pipeline(steps=[
        ('transformer', basic_transformer),
        ('model', LinearRegression())
    ])
    
    # Pipeline 2: Alternative Features
    alt_transformer = ColumnTransformer(
        transformers=[
            ('imputer', SimpleImputer(strategy='mean'), FEATURES_ALT)
        ],
        remainder='drop'
    )
    pipeline_alt = Pipeline(steps=[
        ('transformer', alt_transformer),
        ('model', LinearRegression())
    ])
    
    # Pipeline 3: Extended Features
    ext_transformer = ColumnTransformer(
        transformers=[
            ('imputer', SimpleImputer(strategy='mean'), FEATURES_EXT)
        ],
        remainder='drop'
    )
    pipeline_ext = Pipeline(steps=[
        ('transformer', ext_transformer),
        ('model', LinearRegression())
    ])
    
    # 4. Create the Stacking Ensemble Model (as requested in 2c)
    estimators = [
        ('basic_model', pipeline_basic),
        ('alternative_model', pipeline_alt),
        ('extended_model', pipeline_ext)
    ]
    
    # The meta-model is a simple LinearRegression that learns how to combine
    # the predictions from the three base models.
    stacking_regressor = StackingRegressor(
        estimators=estimators,
        final_estimator=LinearRegression(),
        cv=5,  # Use 5-fold cross-validation to generate base model predictions
        passthrough=False # Only use predictions from base models as features for meta-model
    )
    
    # 5. Train the model
    print("Training the stacking model... (This may take a moment)")
    stacking_regressor.fit(X_train, y_train)
    
    # Store the trained model globally
    STACKING_MODEL = stacking_regressor
    
    print("Model training complete.")
    
    # 6. Evaluate on test set (optional, but good practice)
    score = STACKING_MODEL.score(X_test, y_test)
    print(f"Model R-squared score on test data: {score:.4f}")
    
    # Show the weights (coefficients) the meta-model learned for each base model
    meta_model = STACKING_MODEL.final_estimator_
    print("\nMeta-model weights (coefficients):")
    for (name, _), coef in zip(estimators, meta_model.coef_):
        print(f"  - {name}: {coef:.4f}")
    print(f"  - Intercept: {meta_model.intercept_:.4f}")


def predict_housing_price(features_dict):
    """
    Predicts a housing price given a dictionary of features.
    Any missing features will be automatically imputed by the model's pipelines.
    
    Args:
        features_dict (dict): A dictionary where keys are feature names
                              (e.g., 'square_feet') and values are the feature values.
                              
    Returns:
        float: The predicted housing price, or None if the model isn't trained.
    """
    if STACKING_MODEL is None or ALL_FEATURES is None:
        print("Error: Model is not trained. Please run train_model() first.")
        return None
        
    # 1. Create a single-row DataFrame from the input dictionary.
    # Crucially, we specify 'columns=ALL_FEATURES' to ensure the DataFrame
    # has all the columns the model expects, in the correct order.
    # Features *not* in features_dict will be filled with NaN.
    input_df = pd.DataFrame([features_dict], columns=ALL_FEATURES)
    
    # 2. Use the trained stacking model to predict.
    # Each base pipeline will receive this full DataFrame,
    # select its required columns, impute the NaNs, and make a prediction.
    # The meta-model will then combine these predictions.
    try:
        prediction = STACKING_MODEL.predict(input_df)
        return prediction[0]
    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        return None

# --- Main execution block ---
if __name__ == "__main__":
    
    # Train the model when the script is run
    train_model()
    
    print("\n" + "="*30)
    print("Running Prediction Examples (2d)")
    print("="*30)
    
    # Example 1: Input with features from 'basic' and 'extended' sets
    example_1_features = {
        'square_feet': 2100,
        'bedrooms': 3,
        'bathrooms': 2.5,
        'garage_spaces': 2
    }
    
    pred_1 = predict_housing_price(example_1_features)
    if pred_1 is not None:
        print(f"\nExample 1 Features: {example_1_features}")
        print(f"Predicted Price: ${pred_1:,.2f}")
        print("  (Note: 'age_years', 'year_built', 'school_rating', etc. were imputed)")

    # Example 2: Input with features from 'alternative' and 'extended' sets
    example_2_features = {
        'square_feet': 1800,
        'has_pool': 1,
        'school_rating': 8,
        'distance_to_city_miles': 15.5
    }
    
    pred_2 = predict_housing_price(example_2_features)
    if pred_2 is not None:
        print(f"\nExample 2 Features: {example_2_features}")
        print(f"Predicted Price: ${pred_2:,.2f}")
        print("  (Note: 'bedrooms', 'bathrooms', 'num_rooms', etc. were imputed)")
