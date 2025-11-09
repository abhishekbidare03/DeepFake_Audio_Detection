import os
import sys
import random
import time
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

class Config:
    DATA_DIR = os.path.join(project_root, 'New_dataset')
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    LR_HEAD = 1e-4
    LR_CONV3 = 3e-5
    LR_CONV2 = 1e-5
    WEIGHT_DECAY = 1e-4
    EPOCHS_STAGE1 = 2  # head-only
    EPOCHS_STAGE2 = 6  # unfreeze conv3
    EPOCHS_STAGE3 = 12  # unfreeze conv2
    EARLY_STOPPING = 5
    TARGET_FRAMES = 128
    AUGMENT_PROB = 0.6


class FullDataset(Dataset):
    def __init__(self, split='training', augment=False):
        self.split_dir = os.path.join(Config.DATA_DIR, split)
        self.samples = []
        self.augment = augment
        for label_dir, label in [('real', 0), ('fake', 1)]:
            p = os.path.join(self.split_dir, label_dir)
            if not os.path.exists(p):
                continue
            for fname in sorted(os.listdir(p)):
                if fname.endswith('.wav'):
                    self.samples.append((os.path.join(p, fname), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        y, sr = load_and_process_audio(path)
        # augmentation on waveform
        if self.augment and random.random() < Config.AUGMENT_PROB:
            y = augment_waveform(y, sr)
        feat = extract_features(y, sr)
        # spec augment occasionally
        if self.augment and random.random() < 0.5:
            feat = spec_augment(feat)
        T = feat.shape[1]
        if T < Config.TARGET_FRAMES:
            feat = np.pad(feat, ((0, 0), (0, Config.TARGET_FRAMES - T)), mode='constant')
        elif T > Config.TARGET_FRAMES:
            feat = feat[:, :Config.TARGET_FRAMES]
        tensor = torch.FloatTensor(feat).unsqueeze(0)
        return tensor, torch.LongTensor([label])[0]


def get_loaders(batch_size=None, num_workers=None):
    batch_size = batch_size or Config.BATCH_SIZE
    num_workers = num_workers if num_workers is not None else Config.NUM_WORKERS
    train_ds = FullDataset('training', augment=True)
    val_ds = FullDataset('validation', augment=False)
    test_ds = FullDataset('testing', augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader


def set_requires_grad_for_names(model, name_substrings, requires_grad):
    for name, param in model.named_parameters():
        param.requires_grad = any(ns in name for ns in name_substrings) and requires_grad or (not any(ns in name for ns in name_substrings) and not requires_grad)


def prepare_optimizer(model, group_lrs):
    # group_lrs: list of (name_substring, lr)
    groups = []
    for substr, lr in group_lrs:
        params = [p for n, p in model.named_parameters() if substr in n and p.requires_grad]
        if params:
            groups.append({'params': params, 'lr': lr})
    # fallback: any remaining params that require grad
    remaining = [p for n, p in model.named_parameters() if p.requires_grad and not any(substr in n for substr, _ in group_lrs)]
    if remaining:
        groups.append({'params': remaining, 'lr': group_lrs[-1][1]})
    optimizer = optim.AdamW(groups, weight_decay=Config.WEIGHT_DECAY)
    return optimizer


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            out = model(features)
            loss = criterion(out, labels)
            loss_sum += loss.item()
            preds = out.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return loss_sum / len(loader), 100.0 * correct / total


def train_stage(model, optimizer, train_loader, val_loader, device, epochs, checkpoint_prefix):
    criterion = nn.CrossEntropyLoss()
    best_val = 0.0
    patience = 0
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        pbar = tqdm(train_loader, desc=f'{checkpoint_prefix} epoch {epoch+1}/{epochs}')
        for features, labels in pbar:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(features)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            preds = out.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            pbar.set_postfix({'loss': running_loss/len(train_loader), 'acc': 100.*correct/total})

        val_loss, val_acc = evaluate(model, val_loader, device)
        print(f'[{checkpoint_prefix}] Epoch {epoch+1} val_loss={val_loss:.4f} val_acc={val_acc:.2f}%')
        scheduler.step(val_loss)

        # Save best
        ckpt_path = os.path.join(project_root, 'models', f'{checkpoint_prefix}_best.pth')
        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), ckpt_path)
            patience = 0
            print('Saved checkpoint:', ckpt_path)
        else:
            patience += 1

        if patience >= Config.EARLY_STOPPING:
            print('Early stopping stage:', checkpoint_prefix)
            break

    return best_val


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    train_loader, val_loader, test_loader = get_loaders()
    print('Train samples:', len(train_loader.dataset))
    print('Val samples:', len(val_loader.dataset))
    print('Test samples:', len(test_loader.dataset))

    model = create_model().to(device)
    prev = os.path.join(project_root, 'models', 'best_model.pth')
    if not os.path.exists(prev):
        raise FileNotFoundError('Previous model not found at models/best_model.pth')
    state = torch.load(prev, map_location=device)
    model.load_state_dict(state)

    # Stage 1: head only
    print('\nStage 1: training head (fc layers) only')
    # freeze everything then unfreeze fc
    for name, param in model.named_parameters():
        param.requires_grad = False
        if 'fc' in name:
            param.requires_grad = True

    optimizer = prepare_optimizer(model, [('fc', Config.LR_HEAD)])
    _ = train_stage(model, optimizer, train_loader, val_loader, device, Config.EPOCHS_STAGE1, 'stage1_head')

    # Stage 2: unfreeze conv3 + fc
    print('\nStage 2: unfreeze conv3 + fc')
    for name, param in model.named_parameters():
        if 'conv3' in name or 'fc' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    optimizer = prepare_optimizer(model, [('conv3', Config.LR_CONV3), ('fc', Config.LR_HEAD)])
    _ = train_stage(model, optimizer, train_loader, val_loader, device, Config.EPOCHS_STAGE2, 'stage2_conv3')

    # Stage 3: unfreeze conv2 + conv3 + fc
    print('\nStage 3: unfreeze conv2 + conv3 + fc')
    for name, param in model.named_parameters():
        if 'conv2' in name or 'conv3' in name or 'fc' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    optimizer = prepare_optimizer(model, [('conv2', Config.LR_CONV2), ('conv3', Config.LR_CONV3), ('fc', Config.LR_HEAD)])
    _ = train_stage(model, optimizer, train_loader, val_loader, device, Config.EPOCHS_STAGE3, 'stage3_conv2')

    # Load best from last stage
    best_path = os.path.join(project_root, 'models', 'stage3_conv2_best.pth')
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))

    # Final evaluation on test set
    test_loss, test_acc = evaluate(model, test_loader, device)
    print(f'\nFinal test accuracy: {test_acc:.2f}%')
    final_path = os.path.join(project_root, 'models', f'finetuned_full_testacc_{test_acc:.2f}.pth')
    torch.save(model.state_dict(), final_path)
    print('Saved final finetuned model to', final_path)


if __name__ == '__main__':
    main()
