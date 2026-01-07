import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn import metrics
import os
import glob
import pathlib
from PIL import Image

# --- CONFIGURATION ---
IMG_SIZE = 256
BATCH_SIZE = 32
EPOCHS = 15  # Increased slightly as data pipeline is faster
DATA_DIR = 'lung_colon_image_set' # Base folder name

print("TensorFlow Version:", tf.__version__)
print("GPU Available:", len(tf.config.list_physical_devices('GPU')) > 0)

# --- 1. DATA DISCOVERY ---
# Recursively find all jpeg images. This handles the split between lung/colon folders automatically.
# We look for any .jpeg file inside the data directory
print(f"Searching for images in {DATA_DIR}...")
all_image_paths = glob.glob(f"{DATA_DIR}/**/*.jpeg", recursive=True)
all_image_paths = [p for p in all_image_paths if "macos" not in p.lower()] # Clean up Mac junk files if any

if len(all_image_paths) == 0:
    raise ValueError(f"No images found in {DATA_DIR}. Please check the folder name and extraction.")

# Extract labels from folder names
# Structure is usually: .../class_name/image.jpeg
path_objects = [pathlib.Path(p) for p in all_image_paths]
labels = [p.parent.name for p in path_objects]

# Get unique classes and map them to integers
classes = sorted(list(set(labels)))
class_to_index = {name: i for i, name in enumerate(classes)}
numeric_labels = [class_to_index[l] for l in labels]

print(f"Found {len(all_image_paths)} images belonging to {len(classes)} classes.")
print(f"Classes: {classes}")

# --- 2. VISUALIZATION (Requested Feature) ---
# This block reproduces the visualization code you liked
print("Generating sample visualizations...")

for cat in classes:
    # Find all images for this specific category
    cat_images = [str(p) for p in path_objects if p.parent.name == cat]
    
    if len(cat_images) > 0:
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'Images for {cat} category', fontsize=20)

        for i in range(3):
            # Pick a random image
            k = np.random.randint(0, len(cat_images))
            img_path = cat_images[k]
            
            try:
                img = np.array(Image.open(img_path))
                ax[i].imshow(img)
                ax[i].axis('off')
                ax[i].set_title(os.path.basename(img_path))
            except Exception as e:
                print(f"Could not load image {img_path}: {e}")

        out_file = f"{cat}_examples.png"
        plt.savefig(out_file, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_file}")

# --- 3. DATA SPLIT & PIPELINE (Memory Fix) ---
# Split paths and labels instead of loading the actual images
train_paths, val_paths, train_labels, val_labels = train_test_split(
    all_image_paths, numeric_labels, test_size=0.2, random_state=2022, stratify=numeric_labels
)

# Function to load and preprocess a single image file
def load_and_preprocess_image(path, label):
    # Read file from disk
    image = tf.io.read_file(path)
    # Decode jpeg
    image = tf.image.decode_jpeg(image, channels=3)
    # Resize
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    # Normalize to [0, 1]
    image = image / 255.0
    return image, label

# Create TensorFlow Datasets
# This creates a pipeline that loads data only when needed (lazy loading)
train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))

# Apply transformations
# AUTOTUNE allows TF to optimize CPU/GPU usage automatically
AUTOTUNE = tf.data.AUTOTUNE

train_ds = (train_ds
            .map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
            .shuffle(buffer_size=1000)
            .batch(BATCH_SIZE)
            .prefetch(buffer_size=AUTOTUNE))

val_ds = (val_ds
          .map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
          .batch(BATCH_SIZE)
          .prefetch(buffer_size=AUTOTUNE))

# --- 4. MODEL DEFINITION ---
num_classes = len(classes)

model = keras.models.Sequential([
    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
    
    layers.Conv2D(filters=32, kernel_size=(5, 5), activation='relu', padding='same'),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.BatchNormalization(),
    
    # Output layer must match number of classes
    layers.Dense(num_classes, activation='softmax')
])

model.summary()

# Save summary to file
with open("model_summary.txt", "w") as f:
    model.summary(print_fn=lambda x: f.write(x + "\n"))

# --- 5. TRAINING ---
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy', # Use sparse because labels are integers, not one-hot
    metrics=['accuracy']
)

# Callbacks
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('val_accuracy') is not None and logs.get('val_accuracy') > 0.95:
            print('\nValidation accuracy reached > 95%, stopping training.')
            self.model.stop_training = True

es = keras.callbacks.EarlyStopping(patience=5, monitor='val_accuracy', restore_best_weights=True)
lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=2, factor=0.5, verbose=1)

print("Starting training...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    verbose=1,
    callbacks=[es, lr, myCallback()]
)

# --- 6. PLOTTING & REPORTING ---
# Plot history
history_df = pd.DataFrame(history.history)
plt.figure(figsize=(10, 6))
history_df[['accuracy', 'val_accuracy']].plot()
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True)
plt.savefig("history_df.png")
plt.close()

# Generate Classification Report
print("Generating classification report...")
Y_val_true = []
Y_val_pred = []

# Iterate over the validation dataset to get predictions
for images, labels in val_ds:
    preds = model.predict(images, verbose=0)
    pred_classes = np.argmax(preds, axis=1)
    
    Y_val_true.extend(labels.numpy())
    Y_val_pred.extend(pred_classes)

report = metrics.classification_report(Y_val_true, Y_val_pred, target_names=classes)
print(report)

with open("classification_report.txt", "w") as f:
    f.write(report)

print("DONE. Check output files.")