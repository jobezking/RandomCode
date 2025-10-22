import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- 1. Load and Preprocess Data ---
# Load the 'tips' dataset from Seaborn
tips = sns.load_dataset("tips")

# Convert categorical data to numbers using one-hot encoding
tips_processed = pd.get_dummies(tips, columns=['sex', 'smoker', 'day', 'time'], drop_first=True)

# Separate features (X) and the target we want to predict (y)
X = tips_processed.drop("tip", axis=1)
y = tips_processed["tip"]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numerical features
scaler = StandardScaler()
# Find the columns to scale (avoiding the one-hot encoded ones)
cols_to_scale = ['total_bill', 'size']
X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
X_test[cols_to_scale] = scaler.transform(X_test[cols_to_scale])

# --- 2. Define the Keras Model ---
# Keras automatically uses the GPU if TensorFlow can find it!
model = keras.Sequential([
    # Input layer shape matches our number of features
    layers.Dense(64, activation='relu', input_shape=[len(X_train.columns)]),
    layers.Dense(64, activation='relu'),
    # Output layer: 1 neuron because we are predicting a single number (the tip)
    layers.Dense(1)
])

# Compile the model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.01),
    loss='mean_squared_error',  # Loss function for regression
    metrics=['mean_absolute_error']
)

model.summary()

# --- 3. Train the Model ---
epochs = 200
print("\nStarting training... (Keras will use the GPU automatically)")

# The model.fit() process will be accelerated by your RTX 4060
history = model.fit(
    X_train, y_train,
    epochs=epochs,
    validation_split=0.2,  # Use part of the training data for validation
    verbose=0  # Set to 1 to see progress per epoch
)

print("Training finished!")

# --- 4. Evaluate the Model ---
loss, mae = model.evaluate(X_test, y_test, verbose=0)
print(f"\nEvaluation on test data:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"(This means the model's tip predictions are, on average, off by ${mae:.2f})")
