import torch
import torch.nn as nn
from pathlib import Path

class AudioDetectionModel:
    def __init__(self, model_path: str):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model(model_path)
        self.model.eval()
    
    def load_model(self, model_path: str):
        """Load your trained model"""
        # Adjust this based on your actual model architecture
        model = YourModelArchitecture()  # Replace with your model class
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        return model
    
    def predict(self, features):
        """Make prediction on audio features"""
        with torch.no_grad():
            features = torch.FloatTensor(features).to(self.device)
            output = self.model(features)
            prediction = torch.sigmoid(output).item()
            return prediction

# Singleton instance
_model_instance = None

def get_model():
    global _model_instance
    if _model_instance is None:
        model_path = Path("models/best_model.pth")
        _model_instance = AudioDetectionModel(str(model_path))
    return _model_instance