import os
import librosa
import numpy as np
from collections import defaultdict
from tqdm import tqdm

def count_files(directory):
    """Count files in each subdirectory"""
    counts = defaultdict(lambda: defaultdict(int))
    for split in ['training', 'validation', 'testing']:
        split_dir = os.path.join(directory, split)
        for label in ['real', 'fake']:
            label_dir = os.path.join(split_dir, label)
            if os.path.exists(label_dir):
                files = [f for f in os.listdir(label_dir) if f.endswith('.wav')]
                counts[split][label] = len(files)
    return counts

def analyze_audio_properties(directory, sample_size=50):
    """Analyze audio properties for a sample of files"""
    properties = defaultdict(list)
    
    # Collect all audio files
    all_files = []
    for split in ['training', 'validation', 'testing']:
        for label in ['real', 'fake']:
            dir_path = os.path.join(directory, split, label)
            if os.path.exists(dir_path):
                files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith('.wav')]
                all_files.extend(files)
    
    # Randomly sample files
    if len(all_files) > sample_size:
        all_files = np.random.choice(all_files, sample_size, replace=False)
    
    print(f"\nAnalyzing {len(all_files)} sample files...")
    for file_path in tqdm(all_files):
        try:
            y, sr = librosa.load(file_path, sr=None)
            duration = librosa.get_duration(y=y, sr=sr)
            properties['sample_rates'].append(sr)
            properties['durations'].append(duration)
            properties['shapes'].append(y.shape[0])
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            
    return properties

def main():
    dataset_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'New_dataset')
    
    # Count files
    counts = count_files(dataset_dir)
    print("\nFile counts per split:")
    print("-" * 40)
    for split, labels in counts.items():
        total = sum(labels.values())
        print(f"{split}:")
        for label, count in labels.items():
            print(f"  {label}: {count} files ({count/total*100:.1f}%)")
        print(f"  total: {total} files")
    
    # Analyze properties
    properties = analyze_audio_properties(dataset_dir)
    
    print("\nAudio properties:")
    print("-" * 40)
    if properties['sample_rates']:
        print(f"Sample rates: {np.unique(properties['sample_rates']).tolist()} Hz")
        print(f"Duration range: {min(properties['durations']):.1f}s - {max(properties['durations']):.1f}s")
        print(f"Average duration: {np.mean(properties['durations']):.1f}s")
    
if __name__ == '__main__':
    main()