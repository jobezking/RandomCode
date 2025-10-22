import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

# --- 1. Set up CUDA device ---
# This is the key step: check if CUDA is available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 2. Load and Preprocess Data ---
# Load the 'iris' dataset from Seaborn
iris = sns.load_dataset("iris")

# Convert species names to numbers (0, 1, 2)
X = iris.drop("species", axis=1).values
y = pd.Categorical(iris["species"]).codes

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --- 3. Convert to PyTorch Tensors ---
# Convert numpy arrays to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# --- 4. Define the Neural Network ---
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(4, 50)  # 4 input features (sepal_length, etc.)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 3)  # 3 output classes (3 flower species)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Create the model and MOVE IT TO THE GPU
model = SimpleNet().to(device)

# --- 5. Train the Model ---
criterion = nn.CrossEntropyLoss() # Loss function for classification
optimizer = optim.Adam(model.parameters(), lr=0.01)
epochs = 100

print("Starting training...")
for epoch in range(epochs):
    # MOVE TRAINING DATA TO THE GPU
    inputs = X_train_tensor.to(device)
    labels = y_train_tensor.to(device)

    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, labels)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

print("Training finished!")

# --- 6. Evaluate the Model ---
# We don't need gradients for evaluation
with torch.no_grad():
    # MOVE TEST DATA TO THE GPU
    inputs = X_test_tensor.to(device)
    labels = y_test_tensor.to(device)

    outputs = model(inputs)

    # Get predictions by finding the class with the highest score
    _, predicted = torch.max(outputs.data, 1)

    # Move predictions back to CPU to use with scikit-learn
    predicted_cpu = predicted.cpu().numpy()
    labels_cpu = labels.cpu().numpy()

    accuracy = accuracy_score(labels_cpu, predicted_cpu)
    print(f'Accuracy on test data: {accuracy * 100:.2f} %')
