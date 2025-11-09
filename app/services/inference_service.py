import os
import sys
import torch
import torch.nn.functional as F
import numpy as np

# Add project root to path (same pattern as train script)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

from app.models.fake_detector import create_model
from app.utils.audio_utils import load_and_process_audio, extract_features

# Inference configuration (match training)
TARGET_FRAMES = 128
N_MELS = 128


def _pad_or_truncate_feature(feature, target_frames=TARGET_FRAMES):
    """Pad or truncate the time axis of a feature matrix to target_frames.
    feature: np.ndarray of shape [n_mels, T]
    Returns: np.ndarray of shape [n_mels, target_frames]
    """
    T = feature.shape[1]
    if T < target_frames:
        pad_width = target_frames - T
        feature = np.pad(feature, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)
    elif T > target_frames:
        feature = feature[:, :target_frames]
    return feature


def preprocess_audio_to_tensor(file_path, target_frames=TARGET_FRAMES):
    """Load audio file, extract log-mel features and return a tensor ready for the model.
    Output tensor shape: [1, 1, n_mels, target_frames]
    """
    y, sr = load_and_process_audio(file_path)
    feature = extract_features(y, sr)  # shape [n_mels, T]
    feature = _pad_or_truncate_feature(feature, target_frames=target_frames)
    tensor = torch.FloatTensor(feature).unsqueeze(0).unsqueeze(0)  # [1, 1, n_mels, target_frames]
    return tensor


def load_model(model_path, device=None):
    """Load model weights into architecture and return model on device.
    If device is None, uses CPU or CUDA if available.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = create_model()
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # map_location ensures compatibility across devices
    state = torch.load(model_path, map_location=device)
    # Try full load first, otherwise load matching keys only (handles final-layer size mismatch)
    try:
        model.load_state_dict(state)
    except Exception:
        model_dict = model.state_dict()
        pretrained = {k: v for k, v in state.items() if k in model_dict and v.size() == model_dict[k].size()}
        model_dict.update(pretrained)
        model.load_state_dict(model_dict)

    model.to(device)
    model.eval()
    return model


def predict(audio_path, model_path=None, device=None):
    """Run inference on a single audio file.
    Returns a dict: {label, class_index, probabilities}
    """
    if model_path is None:
        # default location in repo
        model_path = os.path.join(project_root, 'models', 'best_model.pth')

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = load_model(model_path, device=device)
    tensor = preprocess_audio_to_tensor(audio_path)
    tensor = tensor.to(device)

    with torch.no_grad():
        outputs = model(tensor)

        # Handle both binary-logit (single output) and 2-class logits
        if outputs.dim() == 2 and outputs.size(1) == 1:
            # single logit: use sigmoid; prob_fake = sigmoid(output)
            prob_fake = torch.sigmoid(outputs).cpu().numpy()[0][0]
            prob_real = 1.0 - prob_fake
        else:
            probs = F.softmax(outputs, dim=1).cpu().numpy()[0]
            # assume index 1 => fake, index 0 => real
            if len(probs) >= 2:
                prob_fake = float(probs[1])
                prob_real = float(probs[0])
            else:
                # fallback: treat highest as predicted fake/real
                prob_fake = float(probs.max())
                prob_real = 1.0 - prob_fake

        # Determine label
        label = 'AI-Generated Audio' if prob_fake >= 0.5 else 'Authentic Audio'

    return {
        'label': label,
        'prob_fake': float(prob_fake),
        'prob_real': float(prob_real)
    }


if __name__ == '__main__':
    # Quick local test helper (not required), keep minimal to avoid heavy imports
    import argparse

    parser = argparse.ArgumentParser(description='Quickly run inference on an audio file')
    parser.add_argument('--audio', '-a', required=True, help='Path to audio file (.wav/.flac)')
    parser.add_argument('--model', '-m', required=False, help='Path to model .pth file')
    args = parser.parse_args()

    res = predict(args.audio, model_path=args.model)
    print(f"Prediction: {res['label']} (class {res['class_index']})")
    print(f"Probabilities: {res['probabilities']}")
