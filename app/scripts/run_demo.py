import os
import torch
import torch.nn.functional as F
from app.models.fake_detector import AudioCNN
from app.utils.audio_utils import load_and_process_audio, extract_features

MODEL_PATH = 'models/combined_final_testacc_61.95.pth'
DEMO_DIR = 'Demo_samples'


def predict_file(model, file_path, device):
    y, sr = load_and_process_audio(file_path)
    features = extract_features(y, sr)  # shape (n_mels, time)
    tensor = torch.FloatTensor(features).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,n_mels,time)
    model.eval()
    with torch.no_grad():
        out = model(tensor)
        # If model outputs logits (1-d), apply sigmoid
        if out.shape[-1] == 1:
            prob = torch.sigmoid(out).item()
        else:
            # safety: if model outputs two logits, apply softmax and take class 1 prob
            probs = F.softmax(out, dim=1)
            prob = probs[0,1].item()
    label = 'fake' if prob > 0.5 else 'real'
    return prob, label


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    model = AudioCNN().to(device)
    # load carefully (allow mismatch in final layer)
    state = torch.load(MODEL_PATH, map_location=device)
    model_dict = model.state_dict()
    # prefer loading matching keys
    pretrained = {k: v for k, v in state.items() if k in model_dict and v.size() == model_dict[k].size()}
    model_dict.update(pretrained)
    model.load_state_dict(model_dict)
    print(f'Loaded model from {MODEL_PATH} (loaded {len(pretrained)}/{len(model_dict)} params)')

    files = [f for f in os.listdir(DEMO_DIR) if f.lower().endswith(('.wav', '.flac'))]
    if not files:
        print('No audio files found in', DEMO_DIR)
        return

    results = []
    for f in files:
        path = os.path.join(DEMO_DIR, f)
        try:
            prob, label = predict_file(model, path, device)
            print(f'{f}: {label} (prob_fake={prob:.4f})')
            results.append((f, label, prob))
        except Exception as e:
            print(f'Error processing {f}: {e}')

    # Save results
    out_path = 'results/demo_predictions.csv'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as fh:
        fh.write('file,label,prob_fake\n')
        for r in results:
            fh.write(f'{r[0]},{r[1]},{r[2]:.4f}\n')
    print('Saved results to', out_path)

if __name__ == '__main__':
    main()
