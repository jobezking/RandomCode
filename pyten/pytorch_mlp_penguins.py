# file: torch_penguins_mlp.py
import seaborn as sns
import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
df = sns.load_dataset("penguins").dropna()
df = df[df["species"].notna()]
X = df[["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]].values.astype(np.float32)
y_labels = df["species"].values
classes = sorted(df["species"].unique())
y = np.array([classes.index(lbl) for lbl in y_labels], dtype=np.int64)

# Train/val split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train).astype(np.float32)
X_val = scaler.transform(X_val).astype(np.float32)

# Move to tensors
device = "cuda" if torch.cuda.is_available() else "cpu"
X_train_t = torch.tensor(X_train, device=device)
y_train_t = torch.tensor(y_train, device=device)
X_val_t   = torch.tensor(X_val, device=device)
y_val_t   = torch.tensor(y_val, device=device)

# Simple MLP
model = nn.Sequential(
    nn.Linear(4, 32),
    nn.ReLU(),
    nn.Linear(32, 32),
    nn.ReLU(),
    nn.Linear(32, len(classes))
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training loop
for epoch in range(30):
    model.train()
    logits = model(X_train_t)
    loss = criterion(logits, y_train_t)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = model(X_val_t).argmax(dim=1)
        acc = (preds == y_val_t).float().mean().item()
    if epoch % 5 == 0 or epoch == 29:
        print(f"Epoch {epoch:02d} | loss={loss.item():.4f} | val_acc={acc:.3f}")

print("Classes:", classes)
