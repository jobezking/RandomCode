import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# --- 1. Set up CUDA device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 2. Load Data and Define Transforms ---
# Transforms to normalize the image data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)) # Mean and std dev of MNIST
])

# Download and load the training/test data
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# --- 3. Define the CNN Model ---
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3) # 1 input channel (grayscale)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 5 * 5, 128) # 5x5 is image size after pooling
        self.fc2 = nn.Linear(128, 10) # 10 output classes (digits 0-9)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # x shape: [batch_size, 1, 28, 28]
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        # -> [batch_size, 32, 13, 13]
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        # -> [batch_size, 64, 5, 5]

        # Flatten the tensor
        x = x.view(-1, 64 * 5 * 5)
        # -> [batch_size, 1600]

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# Create the model and MOVE IT TO THE GPU
model = CNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# --- 4. Training Function ---
def train(epoch):
    model.train() # Set model to training mode
    for batch_idx, (data, target) in enumerate(train_loader):
        # MOVE BATCH DATA TO THE GPU
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.6f}')

# --- 5. Test Function ---
def test():
    model.eval() # Set model to evaluation mode
    test_loss = 0
    correct = 0
    with torch.no_grad(): # No gradients needed
        for data, target in test_loader:
            # MOVE BATCH DATA TO THE GPU
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.2f}%)\n')

# --- 6. Run Training and Testing ---
print("Starting training on MNIST...")
num_epochs = 5 # Keep this low for a quick test
for epoch in range(1, num_epochs + 1):
    train(epoch)
    test()

print("MNIST training finished!")
