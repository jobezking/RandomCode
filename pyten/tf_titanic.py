# file: tf_titanic_mlp.py
import seaborn as sns
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("TF version:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices("GPU"))

# Load dataset
df = sns.load_dataset("titanic").dropna(subset=["survived"])
# Select features (numeric + one-hot for a few categoricals)
features_num = ["age", "fare", "sibsp", "parch"]
features_cat = ["sex", "class", "embark_town"]
df = df.dropna(subset=features_num + features_cat)
X_num = df[features_num].values.astype(np.float32)
X_cat = pd.get_dummies(df[features_cat], drop_first=True).values.astype(np.float32)
X = np.hstack([X_num, X_cat]).astype(np.float32)
y = df["survived"].values.astype(np.float32)

# Train/val split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler().fit(X_train[:, :len(features_num)])
X_train[:, :len(features_num)] = scaler.transform(X_train[:, :len(features_num)])
X_val[:, :len(features_num)] = scaler.transform(X_val[:, :len(features_num)])

# Build model
inputs = tf.keras.Input(shape=(X.shape[1],))
x = tf.keras.layers.Dense(64, activation="relu")(inputs)
x = tf.keras.layers.Dense(64, activation="relu")(x)
outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
model = tf.keras.Model(inputs, outputs)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=64,
    verbose=1
)

# Evaluate
loss, acc = model.evaluate(X_val, y_val, verbose=0)
print(f"Validation accuracy: {acc:.3f}")
