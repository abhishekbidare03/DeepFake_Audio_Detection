import os
import random
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm

# project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

from app.models.fake_detector import create_model
from app.utils.audio_utils import load_and_process_audio, extract_features
from app.utils.augmentations import augment_waveform, spec_augment

# Experiment config (small subset, 3 epochs)
TRAIN_SAMPLES = 1000  # total (balanced)
VAL_SAMPLES = 400
EPOCHS = 3
BATCH_SIZE = 32
LR_FC = 1e-4
LR_CONV3 = 3e-5
WEIGHT_DECAY = 1e-4
TARGET_FRAMES = 128


def collect_balanced_samples(split_dir, n_total):
    real_dir = os.path.join(split_dir, 'real')
    fake_dir = os.path.join(split_dir, 'fake')
    real_files = [os.path.join(real_dir, f) for f in os.listdir(real_dir) if f.endswith('.wav')]
    fake_files = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir) if f.endswith('.wav')]

    n_each = n_total // 2
    real_sample = random.sample(real_files, min(n_each, len(real_files)))
    fake_sample = random.sample(fake_files, min(n_each, len(fake_files)))
    samples = [(p, 0) for p in real_sample] + [(p, 1) for p in fake_sample]
    random.shuffle(samples)
    return samples


class AugmentedSubsetDataset(Dataset):
    def __init__(self, samples, augment=False):
        self.samples = samples
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        y, sr = load_and_process_audio(path)
        if self.augment:
            y = augment_waveform(y, sr)
        feat = extract_features(y, sr)
        # optional spec augment
        if self.augment and random.random() < 0.5:
            feat = spec_augment(feat)
        T = feat.shape[1]
        if T < TARGET_FRAMES:
            feat = np.pad(feat, ((0, 0), (0, TARGET_FRAMES - T)), mode='constant')
        elif T > TARGET_FRAMES:
            feat = feat[:, :TARGET_FRAMES]
        tensor = torch.FloatTensor(feat).unsqueeze(0)
        return tensor, torch.LongTensor([label])[0]


def prepare_model(device):
    model = create_model().to(device)

    prev_model_path = os.path.join(project_root, 'models', 'best_model.pth')
    if not os.path.exists(prev_model_path):
        raise FileNotFoundError(f'Previous model not found: {prev_model_path}')
    state = torch.load(prev_model_path, map_location=device)
    model.load_state_dict(state)

    # Unfreeze conv3 and all FC layers
    for name, param in model.named_parameters():
        if 'conv3' in name or 'fc' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    # Build param groups
    conv3_params = [p for n, p in model.named_parameters() if 'conv3' in n and p.requires_grad]
    fc_params = [p for n, p in model.named_parameters() if 'fc' in n and p.requires_grad]

    param_groups = [
        {'params': conv3_params, 'lr': LR_CONV3},
        {'params': fc_params, 'lr': LR_FC}
    ]

    optimizer = optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    return model, optimizer, criterion


def validate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            out = model(features)
            preds = out.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return 100.0 * correct / total


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for features, labels in tqdm(loader, desc='Train'):
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(features)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        preds = out.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return total_loss / len(loader), 100.0 * correct / total


def main():
    random.seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    train_samples = collect_balanced_samples(os.path.join(project_root, 'New_dataset', 'training'), TRAIN_SAMPLES)
    val_samples = collect_balanced_samples(os.path.join(project_root, 'New_dataset', 'validation'), VAL_SAMPLES)

    train_ds = AugmentedSubsetDataset(train_samples, augment=True)
    val_ds = AugmentedSubsetDataset(val_samples, augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    print('Train samples:', len(train_ds))
    print('Val samples:', len(val_ds))

    model, optimizer, criterion = prepare_model(device)

    best_val = 0.0
    for epoch in range(EPOCHS):
        print(f'\nEpoch {epoch+1}/{EPOCHS}')
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_acc = validate(model, val_loader, device)
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Acc: {val_acc:.2f}%')

        # Save best
        if val_acc > best_val:
            best_val = val_acc
            out_path = os.path.join(project_root, 'models', 'finetune_exp_best.pth')
            torch.save(model.state_dict(), out_path)
            print('Saved best finetune model to', out_path)

    # Save final
    final_path = os.path.join(project_root, 'models', f'finetune_exp_final_valacc_{best_val:.2f}.pth')
    torch.save(model.state_dict(), final_path)
    print('Saved final model to', final_path)


if __name__ == '__main__':
    main()
