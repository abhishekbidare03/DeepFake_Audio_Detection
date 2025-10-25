from cnn_model import AudioCNN
import torch

model = AudioCNN()
model.load_state_dict(torch.load("app/models/model.pt"))
model.eval()
print("✅ Model loaded successfully!")
