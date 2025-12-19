import tensorflow as tf
from tensorflow import keras
from keras import layers, mixed_precision
import matplotlib.pyplot as plt
import pandas as pd
import os

'''
the program is dynamic. The only requirement is that the dataset directory follows the below.
Each subfolder under root is a class, and the names are used as labels.
root/
   class_A/
      img1.jpg
      img2.jpg
   class_B/
      img3.jpg
      img4.jpg

'''
mixed_precision.set_global_policy("mixed_float16")
# Paths
base_path = "lung_colon_image_set"
sub_folders = ["lung_image_sets", "colon_image_sets"]

# Combine both lung and colon sets into one dataset directory structure
# Assume each subfolder contains category subfolders (e.g., 'lung_aca', 'colon_aca', etc.)
data_dirs = [os.path.join(base_path, sf) for sf in sub_folders]

# Parameters
IMG_SIZE = 256
#BATCH_SIZE = 32
#IMG_SIZE = 512
BATCH_SIZE = 8     
EPOCHS = 10
SPLIT = 0.2

# --- Step 1: Build datasets directly from directory ---
train_datasets = []
val_datasets = []

for data_dir in data_dirs:
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=SPLIT,
        subset="training",
        seed=123,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=SPLIT,
        subset="validation",
        seed=123,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE
    )
    train_datasets.append(train_ds)
    val_datasets.append(val_ds)

# Concatenate lung + colon datasets
train_ds = train_datasets[0].concatenate(train_datasets[1])
val_ds = val_datasets[0].concatenate(val_datasets[1])

# --- Step 2: Prefetch and cache for performance ---
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# --- Step 3: Define CNN model ---
num_classes = len(train_ds.class_names)

model = keras.models.Sequential([
    layers.Rescaling(1./255, input_shape=(IMG_SIZE, IMG_SIZE, 3)),
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

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# --- Step 4: Train with callbacks ---
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

es = EarlyStopping(patience=3, monitor='val_accuracy', restore_best_weights=True)
lr = ReduceLROnPlateau(monitor='val_loss', patience=2, factor=0.5, verbose=1)

history = model.fit(train_ds,
                    validation_data=val_ds,
                    epochs=EPOCHS,
                    callbacks=[es, lr])

# --- Step 5: Plot training history ---
history_df = pd.DataFrame(history.history)
history_df.loc[:, ['accuracy', 'val_accuracy']].plot()
plt.title('Training Accuracy vs Validation Accuracy')
plt.savefig("history_df.png")
plt.close()
