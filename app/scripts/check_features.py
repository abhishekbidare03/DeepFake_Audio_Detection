import os
import numpy as np

def check_features():
    features_dir = "data/features"
    real_dir = os.path.join(features_dir, "real")
    fake_dir = os.path.join(features_dir, "fake")
    
    # Check directories
    print("\n=== Directory Structure ===")
    print(f"Features dir exists: {os.path.exists(features_dir)}")
    print(f"Real dir exists: {os.path.exists(real_dir)}")
    print(f"Fake dir exists: {os.path.exists(fake_dir)}")
    
    # Count files
    print("\n=== File Counts ===")
    if os.path.exists(real_dir):
        real_files = [f for f in os.listdir(real_dir) if f.endswith('.npy')]
        print(f"Real samples: {len(real_files)}")
    else:
        print("Real directory not found")
        
    if os.path.exists(fake_dir):
        fake_files = [f for f in os.listdir(fake_dir) if f.endswith('.npy')]
        print(f"Fake samples: {len(fake_files)}")
    else:
        print("Fake directory not found")
    
    # Check sample dimensions
    print("\n=== Feature Dimensions ===")
    all_shapes = []
    
    for directory in [real_dir, fake_dir]:
        if os.path.exists(directory):
            for file in os.listdir(directory):
                if file.endswith('.npy'):
                    path = os.path.join(directory, file)
                    try:
                        feat = np.load(path)
                        all_shapes.append(feat.shape)
                    except Exception as e:
                        print(f"Error loading {file}: {str(e)}")
    
    if all_shapes:
        print("Found feature shapes:")
        for shape in set(str(s) for s in all_shapes):
            count = sum(1 for s in all_shapes if str(s) == shape)
            print(f"  {shape}: {count} files")
    else:
        print("No valid .npy files found")

if __name__ == "__main__":
    check_features()