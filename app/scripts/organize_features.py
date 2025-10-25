import os
import shutil
from pathlib import Path

# Define paths
FEATURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "features")
REAL_DIR = os.path.join(FEATURES_DIR, "real")
FAKE_DIR = os.path.join(FEATURES_DIR, "fake")

def organize_features():
    # Create directories if they don't exist
    os.makedirs(REAL_DIR, exist_ok=True)
    os.makedirs(FAKE_DIR, exist_ok=True)

    # Get all .npy files
    npy_files = [f for f in os.listdir(FEATURES_DIR) if f.endswith('.npy')]
    
    real_count = 0
    fake_count = 0

    for file in npy_files:
        source_path = os.path.join(FEATURES_DIR, file)
        
        # Skip if the file is already in a subdirectory
        if os.path.dirname(source_path) in [REAL_DIR, FAKE_DIR]:
            continue

        # Classify based on filename prefix
        # Assuming LA = real voice and PA = fake/synthesized voice
        if file.startswith('LA'):
            dest_path = os.path.join(REAL_DIR, file)
            shutil.move(source_path, dest_path)
            real_count += 1
        elif file.startswith('PA'):
            dest_path = os.path.join(FAKE_DIR, file)
            shutil.move(source_path, dest_path)
            fake_count += 1

    print(f"✅ Organization complete!")
    print(f"📊 Statistics:")
    print(f"   - Real samples: {real_count}")
    print(f"   - Fake samples: {fake_count}")
    print(f"   - Total files processed: {real_count + fake_count}")

if __name__ == "__main__":
    organize_features()