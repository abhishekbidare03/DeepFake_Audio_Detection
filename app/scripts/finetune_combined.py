import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from app.models.fake_detector import AudioCNN
from app.utils.audio_utils import AudioDataset
from datetime import datetime

def load_datasets():
    # Load old noisy dataset
    old_train_dataset = AudioDataset("data/train")
    old_val_dataset = AudioDataset("data/val")
    
    # Load new clean dataset
    new_train_dataset = AudioDataset("New_dataset/training")
    new_val_dataset = AudioDataset("New_dataset/validation")
    
    # Combine datasets
    train_dataset = ConcatDataset([old_train_dataset, new_train_dataset])
    val_dataset = ConcatDataset([old_val_dataset, new_val_dataset])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(AudioDataset("New_dataset/testing"), batch_size=32, shuffle=False)
    
    return train_loader, val_loader, test_loader

def train_model(model, train_loader, val_loader, test_loader, device):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3)
    
    best_val_acc = 0
    epochs = 50
    patience = 10
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.float().unsqueeze(1))
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            predictions = (outputs > 0.5).float()
            train_correct += (predictions == labels.unsqueeze(1)).sum().item()
            train_total += labels.size(0)
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels.float().unsqueeze(1))
                
                val_loss += loss.item()
                predictions = (outputs > 0.5).float()
                val_correct += (predictions == labels.unsqueeze(1)).sum().item()
                val_total += labels.size(0)
        
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%')
        
        scheduler.step(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_path = f'models/combined_best_valacc_{val_acc:.2f}.pth'
            torch.save(model.state_dict(), model_path)
            print(f'Saved best model with validation accuracy: {val_acc:.2f}%')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print('Early stopping triggered')
                break
    
    # Final test evaluation
    model.eval()
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predictions = (outputs > 0.5).float()
            test_correct += (predictions == labels.unsqueeze(1)).sum().item()
            test_total += labels.size(0)
    
    test_acc = 100 * test_correct / test_total
    print(f'\nFinal Test Accuracy: {test_acc:.2f}%')
    
    # Save final model with test accuracy
    final_model_path = f'models/combined_final_testacc_{test_acc:.2f}.pth'
    torch.save(model.state_dict(), final_model_path)
    print(f'Saved final model to: {final_model_path}')

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load datasets
    train_loader, val_loader, test_loader = load_datasets()
    
    # Initialize model and load weights from best previous checkpoint
    model = AudioCNN().to(device)
    state_dict = torch.load('models/finetuned_full_testacc_75.37.pth')
    
    # Load all layers except the final classification layer
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and 'fc3' not in k}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    
    # Train model
    train_model(model, train_loader, val_loader, test_loader, device)

if __name__ == '__main__':
    main()