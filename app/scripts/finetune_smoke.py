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

# Smoke test config
NUM_SAMPLES = 500  # total samples from training (balanced)
VAL_SAMPLES = 200  # validation samples to evaluate after 1 epoch
LR = 1e-5
EPOCHS = 1
BATCH_SIZE = 32
TARGET_FRAMES = 128


def collect_balanced_samples(split_dir, n_total):
    """Collect n_total samples balanced across real/fake from split_dir."""
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


class SubsetAudioDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        y, sr = load_and_process_audio(path)
        feat = extract_features(y, sr)
        T = feat.shape[1]
        if T < TARGET_FRAMES:
            feat = np.pad(feat, ((0, 0), (0, TARGET_FRAMES - T)), mode='constant')
        elif T > TARGET_FRAMES:
            feat = feat[:, :TARGET_FRAMES]
        tensor = torch.FloatTensor(feat).unsqueeze(0)
        return tensor, torch.LongTensor([label])[0]


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


def main():
    random.seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    training_dir = os.path.join(project_root, 'New_dataset', 'training')
    validation_dir = os.path.join(project_root, 'New_dataset', 'validation')

    train_samples = collect_balanced_samples(training_dir, NUM_SAMPLES)
    val_samples = collect_balanced_samples(validation_dir, VAL_SAMPLES)

    train_ds = SubsetAudioDataset(train_samples)
    val_ds = SubsetAudioDataset(val_samples)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print('Train samples (smoke):', len(train_ds))
    print('Val samples (smoke):', len(val_ds))

    model = create_model().to(device)

    # load previous best model weights
    prev_model_path = os.path.join(project_root, 'models', 'best_model.pth')
    if not os.path.exists(prev_model_path):
        raise FileNotFoundError(f'Previous model not found at {prev_model_path}')

    state = torch.load(prev_model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)

    # Freeze backbone conv layers
    for name, param in model.named_parameters():
        if 'fc' not in name:
            param.requires_grad = False

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    criterion = nn.CrossEntropyLoss()

    # Single epoch fine-tune
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        correct = 0
        total = 0
        for features, labels in tqdm(train_loader, desc='Finetune (smoke)'):
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
        print(f'Epoch {epoch+1}: loss={total_loss/len(train_loader):.4f}, acc={100.*correct/total:.2f}%')

    val_acc = validate(model, val_loader, device)
    print(f'Validation accuracy after smoke fine-tune: {val_acc:.2f}%')

    # Save interim finetuned weights
    out_path = os.path.join(project_root, 'models', 'smoke_finetuned.pth')
    torch.save(model.state_dict(), out_path)
    print('Saved smoke finetuned model to', out_path)


if __name__ == '__main__':
    main()
