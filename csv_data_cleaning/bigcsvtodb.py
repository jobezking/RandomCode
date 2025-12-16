import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

# 1. Presume the 10 GB CSV already exists
csv_file = "tenGB.csv"

# 2. Load CSV into SQLite database in chunks
conn = sqlite3.connect("tenGB.db")
chunksize = 100_000   # adjust based on memory and speed tradeoff

for chunk in pd.read_csv(csv_file, chunksize=chunksize):
    chunk.to_sql("data", conn, if_exists="append", index=False)
    print(f"Inserted {len(chunk)} rows...")

print("Finished loading CSV into SQLite.")

# 3. Read from SQLite and perform EDA/cleaning
# For EDA, sample a manageable subset
query = "SELECT * FROM data LIMIT 10000"
eda_df = pd.read_sql(query, conn)

print("EDA sample shape:", eda_df.shape)
print("Missing values:\n", eda_df.isnull().sum())
print("Basic stats:\n", eda_df.describe(include="all"))

# Example cleaning
eda_df = eda_df.drop_duplicates()
if "income" in eda_df.columns:
    eda_df["income"] = eda_df["income"].fillna(eda_df["income"].median())

# 4. Train XGBoost model with scikit-learn pipeline
X = eda_df.drop("target", axis=1)
y = eda_df["target"]

numeric_features = ["age", "income"]  # adjust based on your dataset
categorical_features = ["gender", "city"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric="logloss"
    ))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

print("Model trained. Accuracy:", model.score(X_test, y_test))

# 5. Make a prediction with new data
new_data = pd.DataFrame({
    "age": [35],
    "income": [60000],
    "gender": ["F"],
    "city": ["Atlanta"]
})

prediction = model.predict(new_data)
print("Prediction for new data:", prediction)

