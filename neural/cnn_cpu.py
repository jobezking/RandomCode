import os
import time

# --- 1. FORCE CPU ONLY & OPTIMIZE THREADING ---
# Must be at the very top
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn import metrics
import glob
import pathlib
from PIL import Image
from zipfile import ZipFile
from datetime import datetime

data_path = 'lung-and-colon-cancer-histopathological-images.zip'
if os.path.isdir("lung_colon_image_set"):
    print("Directory already exists, skipping extraction.")
else:
    with ZipFile(data_path,'r') as zip:
        zip.extractall()
        print('The data set has been extracted.')

now = datetime.now()
timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
with open("log.txt", mode='a') as file:
    file.write(f"{timestamp}\n")

# Ryzen 9 6900HX Optimization: 8 Cores / 16 Threads
tf.config.threading.set_inter_op_parallelism_threads(8)
tf.config.threading.set_intra_op_parallelism_threads(16)

# --- 2. CONFIGURATION ---
IMG_SIZE = 512  # originally 256
BATCH_SIZE = 16  #originally 32 
EPOCHS = 10
DATA_DIR = 'lung_colon_image_set'

print(f"--- CPU-ONLY TRAINING MODE ---")
print(f"Logic Cores detected: {os.cpu_count()}")

# --- 3. DATA DISCOVERY ---
all_image_paths = glob.glob(f"{DATA_DIR}/**/*.jpeg", recursive=True)
path_objects = [pathlib.Path(p) for p in all_image_paths if "macos" not in p.lower()]
labels = [p.parent.name for p in path_objects]
classes = sorted(list(set(labels)))
class_to_index = {name: i for i, name in enumerate(classes)}
numeric_labels = [class_to_index[l] for l in labels]

print(f"Found {len(all_image_paths)} images across {len(classes)} classes.")

# --- 4. VISUALIZATION (As requested from CNN.py) ---
for cat in classes:
    cat_images = [str(p) for p in path_objects if p.parent.name == cat]
    if len(cat_images) > 0:
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'Sample Images: {cat}', fontsize=20)
        for i in range(3):
            k = np.random.randint(0, len(cat_images))
            img = np.array(Image.open(cat_images[k]))
            ax[i].imshow(img)
            ax[i].axis('off')
        plt.savefig(f"{cat}_examples.png", bbox_inches="tight")
        plt.close(fig)

# --- 5. DATA PIPELINE ---
train_paths, val_paths, train_labels, val_labels = train_test_split(
    [str(p) for p in path_objects], numeric_labels, test_size=0.2, random_state=2022, stratify=numeric_labels
)

def load_and_preprocess(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    return image / 255.0, label

train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
train_ds = (train_ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
            .shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE))

val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
val_ds = (val_ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
          .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE))

# --- 6. MODEL ---


#[Image of a Convolutional Neural Network architecture diagram]

model = keras.models.Sequential([
    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(len(classes), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# --- 7. BENCHMARKING CALLBACK ---
class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []
    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()
    def on_epoch_end(self, epoch, logs={}):
        duration = time.time() - self.epoch_time_start
        self.times.append(duration)
        print(f" - Epoch {epoch+1} took {duration:.2f} seconds")

time_callback = TimeHistory()

# --- 8. TRAINING ---
print("\nStarting CPU-only training on Minisforum UM690...")
history = model.fit(
    train_ds, 
    validation_data=val_ds, 
    epochs=EPOCHS, 
    callbacks=[time_callback]
)

# --- 9. FINAL REPORT ---
avg_time = np.mean(time_callback.times)
print(f"\nTraining Complete!")
print(f"Average time per epoch: {avg_time:.2f} seconds")

# Save plots
pd.DataFrame(history.history).plot(figsize=(10,6))
plt.grid(True)
plt.title("Training Metrics (CPU Only)")
plt.savefig("final_training_log.png")

# Classification Report
Y_true, Y_pred = [], []
for imgs, lbls in val_ds:
    Y_true.extend(lbls.numpy())
    Y_pred.extend(np.argmax(model.predict(imgs, verbose=0), axis=1))

print("\nClassification Report:")
print(metrics.classification_report(Y_true, Y_pred, target_names=classes))

now = datetime.now()
timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
with open("log.txt", mode='a') as file:
    file.write(f"{timestamp}\n")

