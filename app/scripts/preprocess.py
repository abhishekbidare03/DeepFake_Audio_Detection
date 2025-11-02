import os
import sys
import numpy as np
from tqdm import tqdm

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

from app.utils.audio_utils import load_and_process_audio, extract_features, save_audio

# Use absolute paths for directories
RAW_DIR = os.path.join(project_root, "data", "raw")
PROC_DIR = os.path.join(project_root, "data", "processed")
FEATURE_DIR = os.path.join(project_root, "data", "features")

# Ensure output directories exist
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROC_DIR, exist_ok=True)
os.makedirs(FEATURE_DIR, exist_ok=True)

# Create real/fake subdirectories
os.makedirs(os.path.join(FEATURE_DIR, "real"), exist_ok=True)
os.makedirs(os.path.join(FEATURE_DIR, "fake"), exist_ok=True)

def process_all_audios():
    # Get both .wav and .flac files
    files = [f for f in os.listdir(RAW_DIR) if f.endswith((".wav", ".flac"))]
    
    print(f"Found {len(files)} audio files ({sum(f.endswith('.wav') for f in files)} WAV, {sum(f.endswith('.flac') for f in files)} FLAC)")
    print("Starting preprocessing...")
    
    # Calculate total size of files to process
    total_size = sum(os.path.getsize(os.path.join(RAW_DIR, f)) for f in files)
    print(f"Total data size: {total_size / (1024**3):.2f} GB")
    
    # Track progress by file type
    la_files = [f for f in files if f.startswith("LA")]
    pa_files = [f for f in files if f.startswith("PA")]
    print(f"Distribution - Real (LA): {len(la_files)}, Fake (PA): {len(pa_files)}")

    # Process files in batches to manage memory
    batch_size = 1000
    total_processed = 0
    error_count = 0

    total_batches = (len(files) + batch_size - 1) // batch_size
    print(f"\nTotal files to process: {len(files)}")
    print(f"Number of batches: {total_batches} (batch size: {batch_size})")
    print(f"Expected total size: {total_size / (1024**3):.2f} GB\n")
    
    for batch_start in range(0, len(files), batch_size):
        current_batch = batch_start//batch_size + 1
        batch_files = files[batch_start:batch_start + batch_size]
        print(f"\nProcessing batch {current_batch}/{total_batches} ({current_batch/total_batches*100:.1f}% complete)")
        print(f"Files {batch_start} to {min(batch_start + batch_size, len(files))} of {len(files)}")
        
        for file in tqdm(batch_files, desc=f"Batch {current_batch}"):
            path = os.path.join(RAW_DIR, file)
            # Skip if feature already exists
            feature_path = os.path.join(FEATURE_DIR, 
                                      "real" if file.startswith("LA") else "fake", 
                                      os.path.splitext(file)[0] + ".npy")
            if os.path.exists(feature_path):
                continue
            
            try:
                # Step 1: Load and preprocess audio
                y, sr = load_and_process_audio(path)
                
                # Step 2: Save cleaned audio (always save as WAV for processed files)
                proc_path = os.path.join(PROC_DIR, os.path.splitext(file)[0] + ".wav")
                save_audio(y, sr, proc_path)
                
                # Step 3: Extract features
                log_mel = extract_features(y, sr)
                
                # Step 4: Save as .npy in appropriate subdirectory (real/fake)
                subdir = "real" if file.startswith("LA") else "fake"
                feature_path = os.path.join(FEATURE_DIR, subdir, os.path.splitext(file)[0] + ".npy")
                np.save(feature_path, log_mel)
                total_processed += 1
                
            except Exception as e:
                print(f"Error processing {file}: {str(e)}")
                error_count += 1
                continue

        # Free up memory
        if total_processed % 100 == 0:
            import gc
            gc.collect()

    print(f"✅ Preprocessing complete!")
    print(f"Successfully processed: {total_processed} files")
    print(f"Errors encountered: {error_count} files")
    
    # Final distribution check
    final_la = len([f for f in os.listdir(os.path.join(FEATURE_DIR, "real")) if f.endswith(".npy")])
    final_pa = len([f for f in os.listdir(os.path.join(FEATURE_DIR, "fake")) if f.endswith(".npy")])
    print(f"Final distribution - Real (LA): {final_la}, Fake (PA): {final_pa}")

if __name__ == "__main__":
    process_all_audios()
