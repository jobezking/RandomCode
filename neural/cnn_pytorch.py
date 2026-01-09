import os
import glob
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from zipfile import ZipFile
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from sklearn.model_selection import train_test_split
from sklearn import metrics


# -----------------------------
# 0. EXTRACT ZIP IF NEEDED
# -----------------------------
data_path = 'lung-and-colon-cancer-histopathological-images.zip'

if os.path.isdir("lung_colon_image_set"):
    print("Directory already exists, skipping extraction.")
else:
    with ZipFile(data_path, 'r') as zip:
        zip.extractall()
        print("The data set has been extracted.")


# Log timestamp
now = datetime.now()
timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
with open("log.txt", mode='a') as file:
    file.write(f"{timestamp}\n")


# -----------------------------
# 1. CONFIG
# -----------------------------
IMG_SIZE = 512
BATCH_SIZE = 16
EPOCHS = 15
DATA_DIR = "lung_colon_image_set"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("PyTorch Version:", torch.__version__)
print("GPU Available:", torch.cuda.is_available())
print("Device:", device)


# -----------------------------
# 2. DATA DISCOVERY
# -----------------------------
print(f"Searching for images in {DATA_DIR}...")
all_image_paths = glob.glob(f"{DATA_DIR}/**/*.jpeg", recursive=True)
all_image_paths = [p for p in all_image_paths if "macos" not in p.lower()]

if len(all_image_paths) == 0:
    raise ValueError("No images found!")

path_objects = [pathlib.Path(p) for p in all_image_paths]
labels = [p.parent.name for p in path_objects]

classes = sorted(list(set(labels)))
class_to_index = {name: i for i, name in enumerate(classes)}
numeric_labels = [class_to_index[l] for l in labels]

print(f"Found {len(all_image_paths)} images across {len(classes)} classes.")
print("Classes:", classes)


# -----------------------------
# 3. VISUALIZATION
# -----------------------------
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


# -----------------------------
# 4. DATASET CLASS
# -----------------------------
class HistopathDataset(Dataset):
    def __init__(self, paths, labels, img_size):
        self.paths = paths
        self.labels = labels
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),  # Converts to [0,1]
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        label = self.labels[idx]

        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        return img, label


# -----------------------------
# 5. TRAIN/VAL SPLIT
# -----------------------------
train_paths, val_paths, train_labels, val_labels = train_test_split(
    all_image_paths, numeric_labels, test_size=0.2,
    random_state=2022, stratify=numeric_labels
)

train_ds = HistopathDataset(train_paths, train_labels, IMG_SIZE)
val_ds = HistopathDataset(val_paths, val_labels, IMG_SIZE)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)


# -----------------------------
# 6. MODEL (PyTorch CNN)
# -----------------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(128 * (IMG_SIZE // 8) * (IMG_SIZE // 8), 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.BatchNorm1d(128),

            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)


model = SimpleCNN(len(classes)).to(device)
print(model)

with open("model_summary.txt", "w") as f:
    f.write(str(model))


# -----------------------------
# 7. TRAINING LOOP
# -----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

best_val_acc = 0
patience = 5
wait = 0

history_acc = []
history_val_acc = []


print("Starting training...")

for epoch in range(EPOCHS):
    model.train()
    correct = 0
    total = 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), torch.tensor(labels).to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_acc = correct / total

    # Validation
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), torch.tensor(labels).to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_acc = correct / total

    history_acc.append(train_acc)
    history_val_acc.append(val_acc)

    print(f"Epoch {epoch+1}/{EPOCHS} - Train Acc: {train_acc:.4f} - Val Acc: {val_acc:.4f}")

    # Early stopping
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        wait = 0
        torch.save(model.state_dict(), "best_model.pth")
    else:
        wait += 1
        if wait >= patience:
            print("Early stopping triggered.")
            break


# -----------------------------
# 8. PLOT ACCURACY
# -----------------------------
plt.figure(figsize=(10, 6))
plt.plot(history_acc, label="Train Accuracy")
plt.plot(history_val_acc, label="Val Accuracy")
plt.legend()
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid(True)
plt.savefig("history_df.png")
plt.close()


# -----------------------------
# 9. CLASSIFICATION REPORT
# -----------------------------
print("Generating classification report...")

model.load_state_dict(torch.load("best_model.pth"))
model.eval()

Y_true = []
Y_pred = []

with torch.no_grad():
    for imgs, labels in val_loader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        _, preds = torch.max(outputs, 1)

        Y_true.extend(labels)
        Y_pred.extend(preds.cpu().numpy())

report = metrics.classification_report(Y_true, Y_pred, target_names=classes)
print(report)

with open("classification_report.txt", "w") as f:
    f.write(report)

print("DONE. Check output files.")

