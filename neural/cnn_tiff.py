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
from zipfile import ZipFile
from datetime import datetime

# --- DATA EXTRACTION ---
data_path = 'BrainTumorClassification2D.zip'
if os.path.isdir("BrainTumorClassification2D"):
    print("Directory already exists, skipping extraction.")
else:
    with ZipFile(data_path, 'r') as zip:
        zip.extractall()
        print('The data set has been extracted.')

now = datetime.now()
timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
with open("log.txt", mode='a') as file:
    file.write(f"{timestamp}\n")

# --- CONFIGURATION ---
IMG_SIZE = 512 
BATCH_SIZE = 16 
EPOCHS = 15  
DATA_DIR = 'BrainTumorClassification2D' 

print("TensorFlow Version:", tf.__version__)
print("GPU Available:", len(tf.config.list_physical_devices('GPU')) > 0)

# --- 1. DATA DISCOVERY (Updated for TIFF) ---
print(f"Searching for TIFF images in {DATA_DIR}...")
# Updated pattern to catch .tif, .tiff, .TIF, or .TIFF
all_image_paths = glob.glob(f"{DATA_DIR}/**/*.[tT][iI][fF]*", recursive=True)
all_image_paths = [p for p in all_image_paths if "macos" not in p.lower()]

if len(all_image_paths) == 0:
    raise ValueError(f"No TIFF images found in {DATA_DIR}. Please check the folder name.")

path_objects = [pathlib.Path(p) for p in all_image_paths]
labels = [p.parent.name for p in path_objects]
classes = sorted(list(set(labels)))
class_to_index = {name: i for i, name in enumerate(classes)}
numeric_labels = [class_to_index[l] for l in labels]

print(f"Found {len(all_image_paths)} images belonging to {len(classes)} classes.")

# --- 2. VISUALIZATION ---
print("Generating sample visualizations...")
for cat in classes:
    cat_images = [str(p) for p in path_objects if p.parent.name == cat]
    if len(cat_images) > 0:
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'Images for {cat} category', fontsize=20)
        for i in range(3):
            k = np.random.randint(0, len(cat_images))
            img_path = cat_images[k]
            try:
                # PIL handles TIFF naturally for visualization
                img = np.array(Image.open(img_path).convert('RGB'))
                ax[i].imshow(img)
                ax[i].axis('off')
                ax[i].set_title(os.path.basename(img_path))
            except Exception as e:
                print(f"Could not load image {img_path}: {e}")
        plt.savefig(f"{cat}_examples.png", bbox_inches="tight")
        plt.close(fig)

# --- 3. DATA SPLIT & PIPELINE (Updated for TIFF) ---
train_paths, val_paths, train_labels, val_labels = train_test_split(
    all_image_paths, numeric_labels, test_size=0.2, random_state=2022, stratify=numeric_labels
)

def load_and_preprocess_image(path, label):
    # Helper to decode TIFF using PIL
    def _read_tiff(path_tensor):
        p = path_tensor.numpy().decode('utf-8')
        img = Image.open(p).convert('RGB')
        return np.array(img).astype(np.float32)

    # Wrap the python function for the TF Dataset
    [image,] = tf.py_function(_read_tiff, [path], [tf.float32])
    
    # Define shape explicitly (lost during py_function)
    image.set_shape([None, None, 3])
    
    # Standard preprocessing
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = image / 255.0
    return image, label

AUTOTUNE = tf.data.AUTOTUNE
train_ds = (tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
            .map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
            .shuffle(buffer_size=1000)
            .batch(BATCH_SIZE)
            .prefetch(buffer_size=AUTOTUNE))

val_ds = (tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
          .map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
          .batch(BATCH_SIZE)
          .prefetch(buffer_size=AUTOTUNE))

# --- 4. MODEL DEFINITION ---
num_classes = len(classes)
model = keras.models.Sequential([
    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.Conv2D(32, (5, 5), activation='relu', padding='same'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.BatchNormalization(),
    layers.Dense(num_classes, activation='softmax')
])

# --- 5. TRAINING ---
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

es = keras.callbacks.EarlyStopping(patience=5, monitor='val_accuracy', restore_best_weights=True)
lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=2, factor=0.5, verbose=1)

history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=[es, lr])

# --- 6. PLOTTING & REPORTING ---
history_df = pd.DataFrame(history.history)
history_df[['accuracy', 'val_accuracy']].plot()
plt.savefig("history_df.png")

print("Generating classification report...")
Y_val_true, Y_val_pred = [], []
for images, labels in val_ds:
    preds = model.predict(images, verbose=0)
    Y_val_true.extend(labels.numpy())
    Y_val_pred.extend(np.argmax(preds, axis=1))

report = metrics.classification_report(Y_val_true, Y_val_pred, target_names=classes)
with open("classification_report.txt", "w") as f:
    f.write(report)

print("DONE.")