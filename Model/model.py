import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import os
from PIL import Image

train_transforms = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

data = datasets.ImageFolder('/kaggle/input/mrl-dataset/train')

train_size = int(0.8 * len(data))
test_size = len(data) - train_size
seed = torch.Generator().manual_seed(42)
train_dataset, test_dataset = random_split(data, [train_size, test_size], generator=seed)

train_dataset.dataset.transform = train_transforms
test_dataset.dataset.transform = test_transform

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class DDDM(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.c1 = nn.Conv2d(3, 32, 3)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(32)
        self.p1 = nn.MaxPool2d(kernel_size=3)
        self.c2 = nn.Conv2d(32, 64, 3)
        self.bn2 = nn.BatchNorm2d(64)
        self.p2 = nn.MaxPool2d(kernel_size=3)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=64 * 27 * 27, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, X):
        X = self.c1(X)
        X = self.bn1(X)
        X = self.relu(X)
        X = self.p1(X)

        X = self.c2(X)
        X = self.bn2(X)
        X = self.relu(X)
        X = self.p2(X)

        X = self.flatten(X)
        X = self.fc1(X)
        X = self.relu(X)
        X = self.fc2(X)
        return X

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DDDM(num_classes=len(data.classes)).to(device)
criterion = nn.CrossEntropyLoss()
learning_rate = 0.001
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
num_epochs = 10

print("Starting training...")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    print(f"Training in epoch {epoch+1}")
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_dataset)
    epoch_accuracy = (correct_predictions / total_samples) * 100
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_accuracy:.2f}%")

print("\nTraining finished!")

model.eval()
test_correct_predictions = 0
test_total_samples = 0
test_loss = 0.0

print("\nEvaluating on test set...")
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        test_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        test_total_samples += labels.size(0)
        test_correct_predictions += (predicted == labels).sum().item()

test_accuracy = (test_correct_predictions / test_total_samples) * 100
test_avg_loss = test_loss / len(test_dataset)

print(f"Test Loss: {test_avg_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
