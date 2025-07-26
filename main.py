from pp import load_and_preprocess
from dataset import GestureDataset

import torch
from torch.utils.data import DataLoader
from model import CNNLSTMModel
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

(X_train, y_train), (X_test, y_test), le = load_and_preprocess()

train_dataset = GestureDataset(X_train, y_train)
test_dataset = GestureDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

for X_batch, y_batch in train_loader:
    print("X shape:", X_batch.shape)  # Should be (batch_size, window_len, 6)
    print("y shape:", y_batch.shape)  # Should be (batch_size,)
    break


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNNLSTMModel(input_size=6, num_classes=len(le.classes_)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
EPOCHS = 30
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device).float(), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# Evaluation
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device).float(), y_batch.to(device)
        outputs = model(X_batch)
        _, predicted = torch.max(outputs.data, 1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")

for X_batch, y_batch in test_loader:
    X_batch = X_batch.to(device).float()
    outputs = model(X_batch)
    _, predicted = torch.max(outputs.data, 1)
    preds = le.inverse_transform(predicted.cpu().numpy())
    actual = le.inverse_transform(y_batch.numpy())
    print("Predicted:", preds)
    print("Actual:   ", actual)
    break


torch.save(model.state_dict(), "gesture_model.pth")