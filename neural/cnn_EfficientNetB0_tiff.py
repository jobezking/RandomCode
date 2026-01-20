import numpy as np
import pandas as pd
import matplotlib 
matplotlib.use("Agg") # Disable GUI backend
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
import seaborn as sns

# ============================================================
# 0. GLOBAL CONFIG
# ============================================================

DATA_DIR = "BrainTumorClassification2D"
ZIP_PATH = "BrainTumorClassification2D.zip"
IMG_SIZE = 630
BATCH_SIZE = 16
EPOCHS = 20

# Enable mixed precision for RTX 4060
tf.keras.mixed_precision.set_global_policy("mixed_float16")

print("TensorFlow:", tf.__version__)
print("GPU:", tf.config.list_physical_devices("GPU"))

# ============================================================
# 1. EXTRACT DATASET
# ============================================================

if not os.path.isdir(DATA_DIR):
    with ZipFile(ZIP_PATH, "r") as z:
        z.extractall()
        print("Dataset extracted.")
else:
    print("Dataset already exists.")

# Log timestamp
with open("log.txt", "a") as f:
    f.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S\n"))

# ============================================================
# 2. DISCOVER TIFF IMAGES
# ============================================================

all_image_paths = glob.glob(f"{DATA_DIR}/**/*.[tT][iI][fF]*", recursive=True)
all_image_paths = [p for p in all_image_paths if "macos" not in p.lower()]

if not all_image_paths:
    raise ValueError("No TIFF images found.")

path_objects = [pathlib.Path(p) for p in all_image_paths]
labels = [p.parent.name for p in path_objects]
classes = sorted(list(set(labels)))
class_to_index = {c: i for i, c in enumerate(classes)}
numeric_labels = [class_to_index[l] for l in labels]

print(f"Found {len(all_image_paths)} TIFF images across {len(classes)} classes.")

# ============================================================
# 3. TRAIN/VAL SPLIT
# ============================================================

train_paths, val_paths, train_labels, val_labels = train_test_split(
    all_image_paths,
    numeric_labels,
    test_size=0.2,
    stratify=numeric_labels,
    random_state=2022
)

# ============================================================
# 4. DATA LOADING (TIFF → Tensor)
# ============================================================

def load_tiff(path):
    """
    path arrives as a NumPy scalar (np.bytes_) when using tf.numpy_function.
    It is NOT a Tensor, so .numpy() is invalid.
    """
    if isinstance(path, bytes):
        p = path.decode("utf-8")
    else:
        # np.ndarray((), dtype='|S...') → scalar bytes
        p = path.item().decode("utf-8")

    img = Image.open(p).convert("RGB")
    return np.array(img).astype(np.float32)


def load_and_preprocess(path, label):
    image = tf.numpy_function(load_tiff, [path], tf.float32)
    image.set_shape([None, None, 3])

    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image / 255.0
    return image, label

# ============================================================
# 5. DATA AUGMENTATION
# ============================================================

augment = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.10),
    layers.RandomContrast(0.15),
    layers.RandomZoom(0.10),
])

def augment_fn(image, label):
    return augment(image), label

# ============================================================
# 6. BUILD DATASET PIPELINE
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

train_ds = (
    tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    .shuffle(5000)
    .map(load_and_preprocess, num_parallel_calls=AUTOTUNE)
    .cache()  # cache decoded TIFFs in RAM
    .map(augment_fn, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

val_ds = (
    tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
    .map(load_and_preprocess, num_parallel_calls=AUTOTUNE)
    .cache()
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

# ============================================================
# 7. MODEL: EfficientNetB0 (Pretrained)
# ============================================================

base = keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base.trainable = False  # freeze for transfer learning

model = keras.Sequential([
    base,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(len(classes), activation="softmax", dtype="float32")  # float32 output
])

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ============================================================
# 8. TRAINING
# ============================================================

callbacks = [
    keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.5)
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# ============================================================
# 9. PLOTS
# ============================================================

pd.DataFrame(history.history)[["accuracy", "val_accuracy"]].plot()
plt.savefig("history_accuracy.png")

# ============================================================
# 10. CLASSIFICATION REPORT + CONFUSION MATRIX
# ============================================================

Y_true, Y_pred = [], []

for images, labels in val_ds:
    preds = model.predict(images, verbose=0)
    Y_true.extend(labels.numpy())
    Y_pred.extend(np.argmax(preds, axis=1))

report = metrics.classification_report(Y_true, Y_pred, target_names=classes)
with open("classification_report.txt", "w") as f:
    f.write(report)

cm = metrics.confusion_matrix(Y_true, Y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=classes, yticklabels=classes)
plt.savefig("confusion_matrix.png")

print("DONE.")
