import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import wandb  # For experiment tracking

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
import sys
sys.path.append(project_root)

from app.models.fake_detector import create_model

# Configuration
class Config:
    DATA_DIR = os.path.join(project_root, "data", "features")
    BATCH_SIZE = 32  # Increased for faster training
    NUM_WORKERS = 4  # Parallel data loading
    LEARNING_RATE = 3e-4  # Using standard learning rate
    EPOCHS = 100
    EARLY_STOPPING_PATIENCE = 5
    VALIDATION_SPLIT = 0.15
    TEST_SPLIT = 0.15
    TARGET_FRAMES = 128  # fixed time-dimension for feature tensors (pad/truncate to this)

# Custom Dataset
class AudioFeaturesDataset(Dataset):
    def __init__(self, feature_paths, labels):
        self.feature_paths = feature_paths
        self.labels = labels
        
    def __len__(self):
        return len(self.feature_paths)
    
    def __getitem__(self, idx):
        feature = np.load(self.feature_paths[idx])
        # feature is [freq_bins, time_frames], e.g. (128, T)
        # Pad or truncate time dimension to Config.TARGET_FRAMES
        T = feature.shape[1]
        target = Config.TARGET_FRAMES
        if T < target:
            pad_width = target - T
            feature = np.pad(feature, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)
        elif T > target:
            feature = feature[:, :target]

        feature = torch.FloatTensor(feature).unsqueeze(0)  # Add channel dimension -> [1, freq, target]
        label = torch.LongTensor([self.labels[idx]])[0]
        return feature, label

def get_data_loaders():
    # Get all feature paths and labels
    real_dir = os.path.join(Config.DATA_DIR, "real")
    fake_dir = os.path.join(Config.DATA_DIR, "fake")
    
    real_features = [(os.path.join(real_dir, f), 0) for f in os.listdir(real_dir) if f.endswith('.npy')]
    fake_features = [(os.path.join(fake_dir, f), 1) for f in os.listdir(fake_dir) if f.endswith('.npy')]
    
    all_features = real_features + fake_features
    feature_paths, labels = zip(*all_features)
    
    # Split data
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        feature_paths, labels, test_size=Config.TEST_SPLIT, stratify=labels, random_state=42
    )
    
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_paths, train_labels, 
        test_size=Config.VALIDATION_SPLIT/(1-Config.TEST_SPLIT), 
        stratify=train_labels, 
        random_state=42
    )
    
    # Create datasets
    train_dataset = AudioFeaturesDataset(train_paths, train_labels)
    val_dataset = AudioFeaturesDataset(val_paths, val_labels)
    test_dataset = AudioFeaturesDataset(test_paths, test_labels)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=Config.NUM_WORKERS,
        pin_memory=True  # Faster data transfer to GPU
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    progress_bar = tqdm(train_loader, desc='Training')
    for features, labels in progress_bar:
        features, labels = features.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        progress_bar.set_postfix({
            'loss': total_loss/len(train_loader),
            'acc': 100.*correct/total
        })
    
    return total_loss/len(train_loader), 100.*correct/total

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for features, labels in val_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return total_loss/len(val_loader), 100.*correct/total

def main():
    # Initialize wandb
    wandb.init(project="deepfake-audio-detection", name="large-scale-training")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get data loaders
    train_loader, val_loader, test_loader = get_data_loaders()
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model
    model = create_model().to(device)
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    # Some torch versions don't accept the `verbose` kwarg. Omit it for compatibility.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=3
    )
    
    # Training loop
    best_val_acc = 0
    patience_counter = 0
    
    for epoch in range(Config.EPOCHS):
        print(f"\nEpoch {epoch+1}/{Config.EPOCHS}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Log metrics
        wandb.log({
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "learning_rate": optimizer.param_groups[0]['lr']
        })
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(project_root, "models", "best_model.pth"))
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered")
            break
    
    # Final evaluation on test set
    model.load_state_dict(torch.load(os.path.join(project_root, "models", "best_model.pth")))
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f"\nFinal Test Accuracy: {test_acc:.2f}%")
    wandb.log({"test_acc": test_acc})

if __name__ == "__main__":
    main()