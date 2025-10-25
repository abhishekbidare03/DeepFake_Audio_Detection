import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import accuracy_score, f1_score
from cnn_model import AudioCNN
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# ==== CONFIG ====
DATA_DIR = "app/data/features"
EPOCHS = 10
BATCH_SIZE = 16
LR = 0.001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==== DATASET CLASS ====
class AudioDataset(Dataset):
    def __init__(self, root_dir, target_length=48):  # Set target length to max length found
        self.samples = []
        self.target_length = target_length
        
        # Collect all samples
        for label, folder in enumerate(["real", "fake"]):
            folder_path = os.path.join(root_dir, folder)
            if not os.path.exists(folder_path):
                continue
            for file in os.listdir(folder_path):
                if file.endswith(".npy"):
                    self.samples.append((os.path.join(folder_path, file), label))

    def pad_or_trim(self, feat):
        """Pad with zeros or trim the spectrogram to target_length"""
        curr_length = feat.shape[1]
        if curr_length < self.target_length:
            # Pad
            pad_length = self.target_length - curr_length
            feat = np.pad(feat, ((0, 0), (0, pad_length)), mode='constant')
        elif curr_length > self.target_length:
            # Trim from the center
            start = (curr_length - self.target_length) // 2
            feat = feat[:, start:start + self.target_length]
        return feat

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        feat = np.load(path)  # (mel_bins, time)
        feat = self.pad_or_trim(feat)  # Ensure consistent time dimension
        feat = np.expand_dims(feat, axis=0)  # (1, mel_bins, time)
        feat = torch.tensor(feat, dtype=torch.float32)
        return feat, torch.tensor(label, dtype=torch.long)

# ==== LOAD DATA ====
dataset = AudioDataset(DATA_DIR)

# Print dataset statistics
print("\n=== Dataset Statistics ===")
print(f"Total samples: {len(dataset)}")
real_count = sum(1 for _, label in dataset if label == 0)
fake_count = sum(1 for _, label in dataset if label == 1)
print(f"Real samples: {real_count}")
print(f"Fake samples: {fake_count}")

if len(dataset) == 0:
    raise ValueError("No samples found! Make sure you have .npy files in data/features/real and data/features/fake directories")

# Stratified split to maintain class distribution
indices = list(range(len(dataset)))
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size

# Ensure at least one sample in each set
if train_size == 0 or val_size == 0 or test_size == 0:
    raise ValueError(f"Dataset too small to split. Need at least 3 samples, got {len(dataset)}")

train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

print("\n=== Data Split ===")
print(f"Training samples: {len(train_set)}")
print(f"Validation samples: {len(val_set)}")
print(f"Test samples: {len(test_set)}")

train_loader = DataLoader(train_set, batch_size=min(BATCH_SIZE, len(train_set)), shuffle=True)
val_loader = DataLoader(val_set, batch_size=min(BATCH_SIZE, len(val_set)))
test_loader = DataLoader(test_set, batch_size=min(BATCH_SIZE, len(test_set)))

# ==== MODEL ====
model = AudioCNN().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ==== TRAIN LOOP ====
best_val_f1 = 0.0
best_model = None

for epoch in range(EPOCHS):
    # Training
    model.train()
    running_loss = 0.0
    train_preds, train_labels = [], []
    
    for feats, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        feats, labels = feats.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(feats)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Collect predictions
        _, predicted = torch.max(outputs.data, 1)
        train_preds.extend(predicted.cpu().numpy())
        train_labels.extend(labels.cpu().numpy())
    
    avg_loss = running_loss / len(train_loader)
    train_acc = accuracy_score(train_labels, train_preds)
    train_f1 = f1_score(train_labels, train_preds, average='weighted', zero_division=0)

    # Validation
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for feats, labels in val_loader:
            feats, labels = feats.to(DEVICE), labels.to(DEVICE)
            outputs = model(feats)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    print(f"Epoch {epoch+1}: Loss={avg_loss:.4f} | Val_Acc={acc:.4f} | Val_F1={f1:.4f}")

# ==== SAVE MODEL ====
torch.save(model.state_dict(), "app/models/model.pt")
print("✅ Model saved as model.pt")

# ==== TEST EVALUATION ====
model.eval()
test_preds, test_labels = [], []
with torch.no_grad():
    for feats, labels in test_loader:
        feats, labels = feats.to(DEVICE), labels.to(DEVICE)
        outputs = model(feats)
        preds = torch.argmax(outputs, dim=1)
        test_preds.extend(preds.cpu().numpy())
        test_labels.extend(labels.cpu().numpy())

acc = accuracy_score(test_labels, test_preds)
f1 = f1_score(test_labels, test_preds)
print(f"✅ Test Accuracy: {acc:.4f}, Test F1: {f1:.4f}")
