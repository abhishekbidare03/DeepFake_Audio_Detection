import os
import numpy as np

# Load a sample file to check dimensions
data_dir = "data/features"
sample_file = None

# Try both real and fake folders
for folder in ["real", "fake"]:
    folder_path = os.path.join(data_dir, folder)
    if os.path.exists(folder_path):
        files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]
        if files:
            sample_file = os.path.join(folder_path, files[0])
            break

if sample_file:
    feat = np.load(sample_file)
    print(f"Feature shape: {feat.shape}")
else:
    print("No .npy files found in features directory")