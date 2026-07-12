# DeepFake Audio Detection
## Abstract
This repository implements a reproducible pipeline for detecting synthetic (deepfake) speech. The codebase includes data preprocessing, feature extraction, model training and evaluation, and a demo web interface. The project is designed for research: it emphasizes experimental rigor (train/val/test splits, seed control, extensive evaluation metrics), modularity (feature and model choices are modular), and reproducibility (requirements, instructions, and configuration guidance).

Key goals:
- Provide a baseline and extensible framework for deepfake audio detection.
- Encourage reproducible comparisons across feature sets, model families, and evaluation protocols.
- Provide a lightweight demo for qualitative inspection.

## Key Contributions
- End-to-end code for training and evaluating deepfake audio detectors.
- Support for common input representations (waveform, spectrogram, MFCC/LFCC).
- Example server and frontend for rapid demonstration.
- Clear experimental protocol and reproducibility checklist suitable for research publications.

## Repository structure
- `main.py` — entry point for training and experiments.
- `run.py` — inference utilities and single-file evaluation scripts.
- `run_server.py` — script to run the demo server (starts web demo).
- `server.py` — backend API for inference (Flask/FastAPI-based endpoints).
- `requirements.txt` — Python dependencies.
- `data/` — dataset directory (expected structure described below).
- `models/` — saved checkpoints and model definitions (architecture files).
- `results/` — evaluation outputs, logs, and plots.
- `Frontend/` and `frontend_backup/` — demo UI (HTML/CSS/JS).
- `Demo_samples/` — small curated audio samples for quick testing.
- `images/` — figures and visual assets for README or reports.
- `app/` — auxiliary app code (if used for packaging/demo).
- `tests/` — unit/integration tests (if present).

## Experimental overview and recommended protocol
To perform scientifically rigorous experiments, follow these recommendations:

1. Data splits:
   - Use speaker-disjoint train/validation/test splits when possible.
   - Report results on a held-out test set containing unseen speakers and unseen synthesis methods.
   - If using cross-validation, report mean ± std across folds.

2. Metrics:
   - Report classifier accuracy, precision, recall, F1-score.
   - Report Area Under the Receiver Operating Characteristic Curve (AUC-ROC).
   - Report Equal Error Rate (EER) and DET/FAR/FRR at typical operating points (e.g., FPR at 1%).
   - Provide confusion matrices and per-class metrics for deeper insight.

3. Statistical significance:
   - Use bootstrapping or repeated experiments with different random seeds to estimate confidence intervals.

4. Baselines:
   - Compare against naive baselines (logistic regression, random forest) on standard features (MFCCs).
   - Consider comparing to commonly used models in the literature (e.g., ResNet variants on spectrograms, x-vector + classifier).

## Data format and recommended preprocessing
Expected data layout (place audio files under `data/`):
- data/
  - train/
    - real/
    - fake/
  - val/
    - real/
    - fake/
  - test/
    - real/
    - fake/

Recommended preprocessing steps (implemented or easily added to this repo):
- Convert audio to 16 kHz (or standardized sampling rate) mono.
- Apply voice activity detection (VAD) to trim silence, if desired.
- Compute features (choose one or more):
  - Raw waveform (for end-to-end models).
  - Spectrogram / Mel-spectrogram (STFT params: window=25ms, hop=10ms).
  - MFCCs (e.g., 13 or 40 coefficients, delta + delta-delta).
  - LFCCs (robust to some synthesis artifacts).
  - Constant-Q or filterbank energy features where appropriate.
- Normalize features per-utterance or using dataset statistics (mean-variance normalization).

Example feature extraction snippet (conceptual):
```python
import librosa
y, sr = librosa.load(path, sr=16000)
melspec = librosa.feature.melspectrogram(y, sr=sr, n_fft=1024, hop_length=160, n_mels=128)
log_mel = librosa.power_to_db(melspec)
```

## Model architectures (recommended)
The framework supports modular model definitions. Recommended families:
- Convolutional Neural Networks (CNNs) on spectrogram inputs (ResNet-style backbones).
- 1D-CNNs or WaveNet-like blocks for waveform input.
- Transformer-based encoders on frame-level features for long-context modeling.
- Hybrid: x-vector embeddings + fully-connected classifier.

Hyperparameter suggestions:
- Batch size: 16–64 (depends on GPU memory and input length).
- Optimizer: AdamW or Adam with weight decay.
- Learning rate schedule: cosine annealing or ReduceLROnPlateau.
- Loss: binary cross-entropy for binary detection; focal loss if class imbalance exists.
- Regularization: dropout (0.2–0.5), data augmentation (SpecAugment, additive noise).

## Training procedure
General steps:
1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Prepare data in `data/` as described above (update paths in any config or constants).
3. Train:
   ```
   python main.py
   ```
   (If your workflow uses flags or config files, modify the script or pass arguments accordingly.)
4. Checkpoints and logs:
   - Models saved to `models/`.
   - Evaluation outputs and plots saved to `results/`.

Notes:
- Ensure deterministic runs by setting seeds for NumPy/PyTorch/random and reporting them in the experiment logs.
- Use mixed precision training to accelerate training on modern GPUs (if supported).

## Evaluation and reporting
- Use the `run.py` script for batch evaluation of checkpoints on the test set.
- Produce ROC curves, DET curves, and confusion matrices for publication-quality figures (save them under `results/`).
- Log experiment hyperparameters and seeds alongside each result file (`.json` or `.yaml`).

Suggested evaluation code outline:
```python
# pseudocode
model.load_state_dict(checkpoint)
model.eval()
for file in test_files:
    features = extract_features(file)
    score = model.predict(features)
    save_score(file, score)
compute_metrics(scores, labels)
plot_roc(scores, labels, out_path='results/roc.png')
```

## Demo and server
- `run_server.py` and `server.py` provide a lightweight demo interface and API.
- Frontend files in `Frontend/` serve as an example web UI; it calls the backend API to perform inference.
- To try a quick demo (local):
  1. Start the server:
     ```
     python run_server.py
     ```
     or
     ```
     python server.py
     ```
     (use the script appropriate for your chosen backend — examine the files for expected flags/ports).
  2. Open `Frontend/index.html` in a browser, or navigate to the server URL (http://localhost:5000 by default).

Tip: If you expose the server publicly for demos, rate-limit and disable any remote code execution endpoints. Only host sanitized, read-only inference endpoints.

## Reproducibility checklist
When preparing results for publication, include:
- Exact commit hash used for experiments.
- Python environment (requirements.txt or conda environment.yml).
- Random seeds and details on GPU(s) used.
- Dataset versions and exact split lists (file lists with hashes).
- Full training logs and checkpoint references.

## Results 
Include a short table or summary of the best-performing model and main metrics on a held-out test set. For reproducibility, replace the placeholders below with your actual numbers before publication.

Model | AUC-ROC | EER | Accuracy
--- | ---: | ---: | ---:
ResNet-Mel | 0.95 | 8.2% | 91.3%
X-Vector + MLP | 0.92 | 10.1% | 88.6%

## Ablation studies and analysis suggestions
- Compare MFCC vs. log-mel vs. raw waveform inputs.
- Evaluate sensitivity to lossy codecs (MP3, AAC) and additive noise.
- Test robustness to different synthesis methods (TTS engines, voice conversion).
- Visualize per-frequency patterns in learned filters and inspect which bands the model relies on.
- Perform interpretability analysis (e.g., Grad-CAM on spectrograms).

## Limitations and ethical considerations
- Detection is inherently adversarial; state-of-the-art generation methods may bypass detectors.
- False positives (labeling real audio as fake) have social consequences — always report operating points and expected error rates.
- Use responsibly and consider privacy, consent, and legal requirements when processing audio data.

## How to cite
If you use this code in a paper, please cite the repository:
```
@misc{DeepFakeAudioDetection2026,
  author = {abhishekbidare03},
  title = {DeepFake Audio Detection (code)},
  year = {2026},
  howpublished = {\url{https://github.com/abhishekbidare03/DeepFake_Audio_Detection}}
}
```

Also include citations for any models or feature extractors you compare against (e.g., ResNet, x-vector papers, LFCC/MFCC references).

## License
Specify the license for this repository (e.g., MIT, Apache 2.0). If no license file is present, add `LICENSE` with your preferred choice. Without a license, reuse is legally limited.

## Contact and contributions
- Author: abhishekbidare03 — open to issues and PRs on GitHub.
- Contributions: Pull requests welcome — please include unit tests or reproducible experiment logs for significant changes.

## Appendix — quick commands
Install:
```
pip install -r requirements.txt
```

Train (example):
```
python main.py
```

Evaluate:
```
python run.py --mode eval --checkpoint models/best.pth
```

Run demo server:
```
python run_server.py
# or
python server.py
```

Export results and plots:
- After running experiments, check `results/` for CSVs and plots. Use the saved checkpoint referred to in your logs for reproducibility.

---

If you want, I can:
- Insert actual experiment numbers and plots into this README if you point me to files under `results/` or share the values you want shown.
- Create a `README_research.md` or update `README.md` directly with a commit (specify commit message and branch).
