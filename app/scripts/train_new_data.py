import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import wandb

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
import sys
sys.path.append(project_root)

from app.models.fake_detector import create_model
from app.utils.audio_utils import load_and_process_audio, extract_features

# Configuration
class Config:
    DATA_DIR = os.path.join(project_root, "New_dataset")
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    LEARNING_RATE = 3e-4
    EPOCHS = 50
    EARLY_STOPPING_PATIENCE = 4
    TARGET_FRAMES = 128  # matching previous training

# Custom Dataset for pre-split data structure
class AudioFeaturesDataset(Dataset):
    def __init__(self, split='training'):
        self.split_dir = os.path.join(Config.DATA_DIR, split)
        self.samples = []
        
        # Load real samples
        real_dir = os.path.join(self.split_dir, 'real')
        for fname in os.listdir(real_dir):
            if fname.endswith('.wav'):
                self.samples.append((os.path.join(real_dir, fname), 0))
                
        # Load fake samples
        fake_dir = os.path.join(self.split_dir, 'fake')
        for fname in os.listdir(fake_dir):
            if fname.endswith('.wav'):
                self.samples.append((os.path.join(fake_dir, fname), 1))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        audio_path, label = self.samples[idx]
        
        # Load and process audio
        y, sr = load_and_process_audio(audio_path)
        feature = extract_features(y, sr)
        
        # Pad or truncate time dimension
        T = feature.shape[1]
        target = Config.TARGET_FRAMES
        if T < target:
            pad_width = target - T
            feature = np.pad(feature, ((0, 0), (0, pad_width)), mode='constant')
        elif T > target:
            feature = feature[:, :target]
            
        feature = torch.FloatTensor(feature).unsqueeze(0)  # Add channel dim [1, freq, time]
        label = torch.LongTensor([label])[0]
        return feature, label

def get_data_loaders():
    # Create datasets for each split
    train_dataset = AudioFeaturesDataset(split='training')
    val_dataset = AudioFeaturesDataset(split='validation')
    test_dataset = AudioFeaturesDataset(split='testing')
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
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
    wandb.init(
        project="deepfake-audio-detection",
        name="new-dataset-training",
        config={
            "learning_rate": Config.LEARNING_RATE,
            "batch_size": Config.BATCH_SIZE,
            "epochs": Config.EPOCHS,
            "model": "AudioCNN",
            "dataset": "new_dataset"
        }
    )
    
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
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=3
    )
    
    # Training loop
    best_val_acc = 0
    patience_counter = 0
    
    checkpoint_dir = os.path.join(project_root, "models", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
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
        
        # Save best model and checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Save best model
            torch.save(model.state_dict(), 
                      os.path.join(project_root, "models", "best_model_new.pth"))
            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'train_acc': train_acc
            }, os.path.join(checkpoint_dir, f"best_checkpoint.pth"))
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Regular checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'train_acc': train_acc
            }, os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth"))
        
        # Early stopping
        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered")
            break
    
    # Final evaluation on test set
    model.load_state_dict(torch.load(os.path.join(project_root, "models", "best_model_new.pth")))
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f"\nFinal Test Accuracy: {test_acc:.2f}%")
    wandb.log({"test_acc": test_acc})
    
    # Save final model with timestamp
    final_model_path = os.path.join(project_root, "models", 
                                   f"model_new_data_acc_{test_acc:.2f}.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to: {final_model_path}")

if __name__ == "__main__":
    main()