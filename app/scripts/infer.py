import os
import sys
import traceback


# Match train.py project_root pattern
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

from app.services.inference_service import predict

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Run the trained DeepFake audio detector on a single audio file')
    parser.add_argument('--audio', '-a', required=True, help='Path to audio file (.wav or .flac)')
    parser.add_argument('--model', '-m', required=False, help='Path to model .pth file (default: models/best_model.pth)')
    args = parser.parse_args()

    audio_path = args.audio
    if not os.path.exists(audio_path):
        print(f"Audio file not found: {audio_path}")
        return

    model_path = args.model
    if model_path is None:
        model_path = os.path.join(project_root, 'models', 'best_model.pth')

    print(f"Using model: {model_path}")
    print(f"Running inference on: {audio_path}")

    try:
        res = predict(audio_path, model_path=model_path)
        print('\n=== Inference result ===')
        print(f"Predicted label: {res['label']} (class {res['class_index']})")
        print(f"Class probabilities: {res['probabilities']}")
    except Exception as e:
        print(f"Error during inference: {e}")
        traceback.print_exc()



if __name__ == '__main__':
    main()
