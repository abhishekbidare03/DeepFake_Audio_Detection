import os
import torch
import numpy as np
from collections import Counter
from sklearn.metrics import confusion_matrix

# project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
import sys
sys.path.append(project_root)

from app.models.fake_detector import create_model
from app.utils.audio_utils import load_and_process_audio, extract_features

TARGET_FRAMES = 128


def pad_or_truncate(feature, target=TARGET_FRAMES):
    T = feature.shape[1]
    if T < target:
        pad_width = target - T
        feature = np.pad(feature, ((0, 0), (0, pad_width)), mode='constant')
    elif T > target:
        feature = feature[:, :target]
    return feature


def load_test_files(test_dir):
    files = []
    labels = []
    for label_name, label_idx in [('real', 0), ('fake', 1)]:
        dir_path = os.path.join(test_dir, label_name)
        for fname in sorted(os.listdir(dir_path)):
            if fname.endswith('.wav'):
                files.append(os.path.join(dir_path, fname))
                labels.append(label_idx)
    return files, labels


def evaluate(model_path=None):
    if model_path is None:
        model_path = os.path.join(project_root, 'models', 'model_new_data_acc_49.91.pth')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    test_dir = os.path.join(project_root, 'New_dataset', 'testing')
    files, labels = load_test_files(test_dir)
    y_true = []
    y_pred = []
    errors = []

    for fp, lbl in zip(files, labels):
        try:
            y, sr = load_and_process_audio(fp)
            feat = extract_features(y, sr)
            feat = pad_or_truncate(feat)
            tensor = torch.FloatTensor(feat).unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(tensor)
                pred = int(torch.argmax(out, dim=1).item())
            y_true.append(lbl)
            y_pred.append(pred)
            if pred != lbl and len(errors) < 20:
                errors.append((fp, lbl, pred))
        except Exception as e:
            print(f"Error on {fp}: {e}")

    cm = confusion_matrix(y_true, y_pred)
    acc = np.mean(np.array(y_true) == np.array(y_pred)) * 100.0

    print("\nEvaluation results:")
    print(f"Model: {model_path}")
    print(f"Total samples: {len(y_true)}")
    print(f"Accuracy: {acc:.2f}%")
    print("Confusion matrix:")
    print(cm)

    print("\nSample errors (up to 20):")
    for fp, lbl, pred in errors:
        print(f"File: {fp} -- true: {lbl} pred: {pred}")

    # Save a small report
    report_path = os.path.join(project_root, 'models', 'eval_report_new_model.txt')
    with open(report_path, 'w') as f:
        f.write(f"Model: {model_path}\n")
        f.write(f"Total: {len(y_true)}\n")
        f.write(f"Accuracy: {acc:.2f}%\n")
        f.write("Confusion matrix:\n")
        f.write(np.array2string(cm))
        f.write("\n\nSample errors:\n")
        for fp, lbl, pred in errors:
            f.write(f"{fp} true:{lbl} pred:{pred}\n")

    print(f"Report saved to: {report_path}")


if __name__ == '__main__':
    evaluate()
