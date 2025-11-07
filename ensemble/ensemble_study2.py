import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator
from sklearn.metrics import r2_score
from typing import Dict, Any, List, Tuple


# -----------------------------
# 1) Load datasets
# -----------------------------
def load_datasets(
    path_basic: str,
    path_extended: str,
    path_alternative: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_basic = pd.read_csv(path_basic)
    df_extended = pd.read_csv(path_extended)
    df_alt = pd.read_csv(path_alternative)
    return df_basic, df_extended, df_alt


# -----------------------------
# 2) Build pipelines per dataset
# -----------------------------
def build_linear_pipeline(feature_names: List[str]) -> Pipeline:
    """
    Returns a pipeline with median imputation and linear regression.
    """
    # We rely on column order matching feature_names when passing arrays.
    # For more safety, we’ll handle DataFrame columns in predict/fit helpers.
    pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("linreg", LinearRegression())
    ])
    return pipeline


def get_feature_sets():
    """
    Define the per-dataset feature names (X columns) and label.
    """
    label = "housing_price"
    features_basic = ["square_feet", "bedrooms", "bathrooms", "age_years"]
    features_extended = [
        "square_feet", "bedrooms", "bathrooms",
        "garage_spaces", "lot_size_sqft", "distance_to_city_miles"
    ]
    features_alt = ["square_feet", "year_built", "num_rooms", "has_pool", "school_rating"]
    return (features_basic, features_extended, features_alt), label


# -----------------------------
# 3) Fit base models
# -----------------------------
def fit_base_models(
    df_basic: pd.DataFrame,
    df_extended: pd.DataFrame,
    df_alt: pd.DataFrame
):
    (features_basic, features_extended, features_alt), label = get_feature_sets()

    # Create pipelines
    pipe_basic = build_linear_pipeline(features_basic)
    pipe_extended = build_linear_pipeline(features_extended)
    pipe_alt = build_linear_pipeline(features_alt)

    # Fit pipelines
    pipe_basic.fit(df_basic[features_basic], df_basic[label])
    pipe_extended.fit(df_extended[features_extended], df_extended[label])
    pipe_alt.fit(df_alt[features_alt], df_alt[label])

    return {
        "basic": {"pipeline": pipe_basic, "features": features_basic},
        "extended": {"pipeline": pipe_extended, "features": features_extended},
        "alternative": {"pipeline": pipe_alt, "features": features_alt},
        "label": label
    }


# -----------------------------
# 4) Train stacking meta-model
# -----------------------------
def build_meta_training_frame(
    models: Dict[str, Any],
    dfs: List[pd.DataFrame]
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Construct a meta-dataset:
      - For each row in each source DataFrame, compute predictions from all base models.
      - Use the row's true housing_price as the target.
    This yields broad coverage across different feature spaces, allowing the meta-model
    to learn weights that generalize across scenarios.
    """
    label = models["label"]
    base_keys = ["basic", "extended", "alternative"]

    meta_rows = []
    meta_targets = []

    for df in dfs:
        # Prepare inputs for each base model from the current df
        preds_for_df = {}

        for key in base_keys:
            feats = models[key]["features"]
            pipe = models[key]["pipeline"]

            # Build a compatible input frame: take matching columns from df.
            # If any required columns are missing, create them as NaN to be imputed.
            X_compat = pd.DataFrame(index=df.index, columns=feats, dtype=float)
            for c in feats:
                if c in df.columns:
                    X_compat[c] = df[c]
                else:
                    X_compat[c] = np.nan  # will be imputed by the pipeline

            # Predict with the base model
            preds_for_df[key] = pipe.predict(X_compat.values)

        # Stack predictions and targets
        combined = pd.DataFrame({
            "pred_basic": preds_for_df["basic"],
            "pred_extended": preds_for_df["extended"],
            "pred_alternative": preds_for_df["alternative"]
        }, index=df.index)

        meta_rows.append(combined)
        meta_targets.append(df[label])

    # Concatenate meta features and targets across all source datasets
    X_meta = pd.concat(meta_rows, axis=0).reset_index(drop=True)
    y_meta = pd.concat(meta_targets, axis=0).reset_index(drop=True)

    return X_meta, y_meta


def fit_meta_model(X_meta: pd.DataFrame, y_meta: pd.Series) -> LinearRegression:
    """
    A simple linear regression meta-model that learns how to weight the base predictions.
    """
    meta_reg = LinearRegression()
    meta_reg.fit(X_meta, y_meta)
    return meta_reg


# -----------------------------
# 5) Driver prediction function
# -----------------------------
class HousingStackedEnsemble(BaseEstimator):
    """
    Wrapper that holds base models and the meta-model.
    Provides:
      - predict_single(input_dict): partial input dict → ensemble price
      - predict_batch(df): batch prediction with partial columns
    """
    def __init__(self, models: Dict[str, Any], meta_model: LinearRegression):
        self.models = models
        self.meta_model = meta_model
        self.base_keys = ["basic", "extended", "alternative"]

    def _prepare_X_for_model(self, input_df: pd.DataFrame, key: str) -> np.ndarray:
        feats = self.models[key]["features"]
        # Align columns in order; missing features become NaN
        X_compat = pd.DataFrame(columns=feats, index=input_df.index, dtype=float)
        for c in feats:
            X_compat[c] = input_df[c] if c in input_df.columns else np.nan
        return X_compat.values

    def predict_batch(self, input_df: pd.DataFrame) -> np.ndarray:
        """
        Accepts a DataFrame with any subset of union(features).
        Missing base-specific features are imputed within each pipeline.
        Returns ensemble predictions.
        """
        # Get base predictions
        base_preds = {}
        for key in self.base_keys:
            pipe = self.models[key]["pipeline"]
            X_arr = self._prepare_X_for_model(input_df, key)
            base_preds[key] = pipe.predict(X_arr)

        # Form meta features [pred_basic, pred_extended, pred_alternative]
        X_meta = pd.DataFrame({
            "pred_basic": base_preds["basic"],
            "pred_extended": base_preds["extended"],
            "pred_alternative": base_preds["alternative"]
        }, index=input_df.index)

        # Ensemble prediction
        return self.meta_model.predict(X_meta)

    def predict_single(self, input_dict: Dict[str, Any]) -> float:
        """
        Single-row prediction from a partial feature dict.
        """
        # Convert to single-row DataFrame
        input_df = pd.DataFrame([input_dict])
        pred = self.predict_batch(input_df)[0]
        return float(pred)


# -----------------------------
# 6) End-to-end build function
# -----------------------------
def build_and_train_all(
    path_basic: str,
    path_extended: str,
    path_alternative: str
) -> HousingStackedEnsemble:
    # Load data
    df_basic, df_extended, df_alt = load_datasets(path_basic, path_extended, path_alternative)

    # Fit base models
    models = fit_base_models(df_basic, df_extended, df_alt)

    # Build meta training set using all rows from all datasets
    X_meta, y_meta = build_meta_training_frame(models, [df_basic, df_extended, df_alt])

    # Fit meta-model
    meta_model = fit_meta_model(X_meta, y_meta)

    # Optionally, you can inspect meta-model weights:
    coef_info = pd.Series(meta_model.coef_, index=X_meta.columns)
    intercept = meta_model.intercept_
    print("Meta-model weights:")
    print(coef_info)
    print(f"Meta-model intercept: {intercept:,.2f}")

    # Wrap in ensemble
    ensemble = HousingStackedEnsemble(models, meta_model)
    return ensemble


# -----------------------------
# 7) Example usage
# -----------------------------
if __name__ == "__main__":
    # Paths to your CSV files
    path_basic = "housing_prices_basic.csv"
    path_extended = "housing_prices_extended.csv"
    path_alternative = "housing_prices_alternative.csv"

    # Train everything
    ensemble = build_and_train_all(path_basic, path_extended, path_alternative)

    # Example 1: Mix of features across datasets (some missing)
    input_example_1 = {
        # Basic features
        "square_feet": 2400,
        "bedrooms": 4,
        # bathrooms missing → imputed in basic pipeline
        "age_years": 12,
        # Extended-only features (some provided, some missing)
        "garage_spaces": 2,
        # lot_size_sqft missing → imputed in extended pipeline
        "distance_to_city_miles": 8.0,
        # Alternative-only features (some provided, some missing)
        "year_built": 2012,
        "num_rooms": 8,
        "has_pool": 0,
        "school_rating": 7
    }
    pred_1 = ensemble.predict_single(input_example_1)
    print(f"Predicted housing price (example 1): ${pred_1:,.0f}")

    # Example 2: Sparse input — let imputation do more work
    input_example_2 = {
        "square_feet": 3100,   # present in all three datasets
        # Purposefully omitting most other features
        "bathrooms": 3.0,      # basic & extended
        "year_built": 2005,    # alternative
        "has_pool": 1          # alternative
        # bedrooms, age_years, lot_size_sqft, distance_to_city_miles,
        # garage_spaces, num_rooms, school_rating all missing → imputed.
    }
    pred_2 = ensemble.predict_single(input_example_2)
    print(f"Predicted housing price (example 2): ${pred_2:,.0f}")
