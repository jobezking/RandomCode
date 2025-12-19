import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from glob import glob

from sklearn.model_selection import train_test_split
from sklearn import metrics

from zipfile import ZipFile
import cv2
import gc
import os

import tensorflow as tf
from tensorflow import keras
from keras import layers

from sklearn.preprocessing import OneHotEncoder

import warnings
warnings.filterwarnings('ignore')

data_path = 'lung-and-colon-cancer-histopathological-images.zip'

# 1. Extraction
if not os.path.exists('lung_colon_image_set'):
    with ZipFile(data_path, 'r') as zip_ref:
        zip_ref.extractall()
        print('The data set has been extracted.')

# --- ADDRESSING POINT 2: Include both lung and colon sets ---
base_path = "lung_colon_image_set"
sub_folders = ["lung_image_sets", "colon_image_sets"]

all_image_data = [] # To store (path, category) tuples
classes = []

for folder in sub_folders:
    folder_path = os.path.join(base_path, folder)
    if os.path.exists(folder_path):
        # Identify categories in each sub-folder
        categories = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
        for cat in categories:
            classes.append(cat)
            image_dir = os.path.join(folder_path, cat)
            print(f"Loading images from: {image_dir}")
            
            # 4. ADDRESSING POINT 4: Correct figure handling in loops
            images = os.listdir(image_dir)
            if images:
                fig, ax = plt.subplots(1, 3, figsize=(15, 5))
                fig.suptitle(f'Images for {cat} category', fontsize=20)
                for i in range(3):
                    k = np.random.randint(0, len(images))
                    img_path = os.path.join(image_dir, images[k])
                    img = np.array(Image.open(img_path))
                    ax[i].imshow(img)
                    ax[i].axis('off')
                plt.savefig(f"{cat}_examples.png", bbox_inches="tight")
                plt.close(fig) # Explicitly close the specific figure object

            # Collect paths for data loading
            category_images = glob(os.path.join(image_dir, '*.jpeg'))
            for img_path in category_images:
                all_image_data.append((img_path, cat))

IMG_SIZE = 256
SPLIT = 0.2
EPOCHS = 10
BATCH_SIZE = 64

X = []
Y = []

# Load and resize images
for img_path, cat in all_image_data:
    img = cv2.imread(img_path)
    X.append(cv2.resize(img, (IMG_SIZE, IMG_SIZE)))
    # Map category string to its index in the classes list
    Y.append(classes.index(cat))

X = np.asarray(X)
Y = np.array(Y)

encoder = OneHotEncoder(sparse_output=False)
one_hot_encoded_Y = encoder.fit_transform(Y.reshape(-1, 1))

X_train, X_val, Y_train, Y_val = train_test_split(X, one_hot_encoded_Y, test_size=SPLIT, random_state=2022)

# --- ADDRESSING POINT 3: Dynamic Output Layer ---
model = keras.models.Sequential([
    layers.Conv2D(filters=32, kernel_size=(5, 5), activation='relu', 
                  input_shape=(IMG_SIZE, IMG_SIZE, 3), padding='same'),
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
    # Automatically set output units to the number of classes found (5)
    layers.Dense(len(classes), activation='softmax') 
])

with open("model_summary.txt", "w") as f:
    model.summary(print_fn=lambda x: f.write(x + "\n"))

from keras.callbacks import EarlyStopping, ReduceLROnPlateau

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('val_accuracy') is not None and logs.get('val_accuracy') > 0.90:
            print('\nValidation accuracy has reached 90%, stopping training.')
            self.model.stop_training = True

es = EarlyStopping(patience=3, monitor='val_accuracy', restore_best_weights=True)
lr = ReduceLROnPlateau(monitor='val_loss', patience=2, factor=0.5, verbose=1)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val),
                    batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1,
                    callbacks=[es, lr, myCallback()])

# 4. ADDRESSING POINT 4: Correct Plotting and Closing
history_df = pd.DataFrame(history.history)
history_plot = history_df.loc[:,['accuracy','val_accuracy']].plot()
plt.title('Training Accuracy vs Validation Accuracy')
plt.savefig("history_df.png")
plt.close() # Closes the current active plot (the history plot)

Y_pred = model.predict(X_val)
Y_val_labels = np.argmax(Y_val, axis=1)
Y_pred_labels = np.argmax(Y_pred, axis=1)

report = metrics.classification_report(Y_val_labels, Y_pred_labels, target_names=classes)
with open("classification_report.txt", "w") as f:
    f.write(report)