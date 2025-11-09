import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm

# project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

from app.models.fake_detector import create_model
from app.utils.audio_utils import load_and_process_audio, extract_features


class Config:
    DATA_DIR = os.path.join(project_root, 'New_dataset')
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    LR = 1e-5
    EPOCHS = 20
    EARLY_STOPPING = 5
    TARGET_FRAMES = 128


class AudioFeaturesDataset(Dataset):
    def __init__(self, split='training'):
        self.split_dir = os.path.join(Config.DATA_DIR, split)
        self.samples = []
        for label_dir, label in [('real', 0), ('fake', 1)]:
            p = os.path.join(self.split_dir, label_dir)
            for fname in sorted(os.listdir(p)):
                if fname.endswith('.wav'):
                    self.samples.append((os.path.join(p, fname), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        y, sr = load_and_process_audio(path)
        feat = extract_features(y, sr)
        T = feat.shape[1]
        if T < Config.TARGET_FRAMES:
            feat = np.pad(feat, ((0, 0), (0, Config.TARGET_FRAMES - T)), mode='constant')
        elif T > Config.TARGET_FRAMES:
            feat = feat[:, :Config.TARGET_FRAMES]
        tensor = torch.FloatTensor(feat).unsqueeze(0)
        return tensor, torch.LongTensor([label])[0]


def get_loaders():
    train_ds = AudioFeaturesDataset('training')
    val_ds = AudioFeaturesDataset('validation')
    test_ds = AudioFeaturesDataset('testing')

    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=Config.NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS)
    return train_loader, val_loader, test_loader


def freeze_backbone(model):
    # Freeze convolutional layers (conv1..conv3)
    for name, param in model.named_parameters():
        if 'fc' not in name:
            param.requires_grad = False


def train(model, train_loader, val_loader, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=Config.LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    best_val = 0
    patience = 0

    for epoch in range(Config.EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for features, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}'):
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()

        val_loss, val_acc = validate(model, val_loader, device)
        print(f'Epoch {epoch+1}: train_loss={total_loss/len(train_loader):.4f} train_acc={100.*correct/total:.2f} val_loss={val_loss:.4f} val_acc={val_acc:.2f}%')
        scheduler.step(val_loss)

        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), os.path.join(project_root, 'models', 'best_model_finetuned.pth'))
            patience = 0
        else:
            patience += 1

        if patience >= Config.EARLY_STOPPING:
            print('Early stopping')
            break


def validate(model, loader, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()
    return total_loss/len(loader), 100.*correct/total


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    train_loader, val_loader, test_loader = get_loaders()
    print('Train samples:', len(train_loader.dataset))

    model = create_model().to(device)

    # Load previous high-performing model
    prev_model_path = os.path.join(project_root, 'models', 'best_model.pth')
    if not os.path.exists(prev_model_path):
        raise FileNotFoundError(f'Previous model not found: {prev_model_path}')

    state = torch.load(prev_model_path, map_location=device)
    model.load_state_dict(state)

    # Freeze backbone and fine-tune FC layers
    freeze_backbone(model)

    train(model, train_loader, val_loader, device)

    # Final evaluation on test set
    model.load_state_dict(torch.load(os.path.join(project_root, 'models', 'best_model_finetuned.pth'), map_location=device))
    test_loss, test_acc = validate(model, test_loader, device)
    print(f'Final finetuned test accuracy: {test_acc:.2f}%')
    torch.save(model.state_dict(), os.path.join(project_root, 'models', f'finetuned_model_testacc_{test_acc:.2f}.pth'))


if __name__ == '__main__':
    main()
