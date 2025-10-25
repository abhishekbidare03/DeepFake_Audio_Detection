import torch
import torch.nn as nn
import torch.nn.functional as F

class AudioCNN(nn.Module):
    def __init__(self):
        super(AudioCNN, self).__init__()
        # Input shape: (batch_size, 1, 128, 48)
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)  # Output: (16, 128, 48)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)  # Output: (32, 64, 24)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)  # Output: (64, 32, 12)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        
        # Calculate the size after convolutions and pooling
        self.fc1 = nn.Linear(64 * 16 * 6, 128)  # Adjusted for 48 time steps
        self.fc2 = nn.Linear(128, 2)  # 2 classes: Real, Fake

    def forward(self, x):
        # Input shape: (batch_size, 1, 128, 42)
        x = self.pool(F.relu(self.conv1(x)))     # (batch_size, 16, 64, 21)
        x = self.pool(F.relu(self.conv2(x)))     # (batch_size, 32, 32, 10)
        x = self.pool(F.relu(self.conv3(x)))     # (batch_size, 64, 16, 5)
        
        x = x.view(x.size(0), -1)                # Flatten: (batch_size, 64 * 16 * 5)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
