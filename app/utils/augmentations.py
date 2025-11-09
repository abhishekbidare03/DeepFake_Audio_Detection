import random
import numpy as np
import librosa


def add_noise(y, snr_db_min=0, snr_db_max=20):
    """Add white gaussian noise to achieve a random SNR between snr_db_min and snr_db_max."""
    snr_db = random.uniform(snr_db_min, snr_db_max)
    rms_signal = np.sqrt(np.mean(y ** 2))
    snr_linear = 10 ** (snr_db / 20.0)
    rms_noise = rms_signal / snr_linear
    noise = np.random.normal(0, rms_noise, size=y.shape)
    return y + noise


def random_time_stretch(y, lower=0.9, upper=1.1):
    rate = random.uniform(lower, upper)
    # librosa.effects.time_stretch requires non-empty signal
    try:
        return librosa.effects.time_stretch(y, rate)
    except Exception:
        return y


def random_pitch_shift(y, sr, n_steps_min=-1, n_steps_max=1):
    n_steps = random.uniform(n_steps_min, n_steps_max)
    try:
        return librosa.effects.pitch_shift(y, sr, n_steps)
    except Exception:
        return y


def spec_augment(feat, time_mask_param=20, freq_mask_param=10, num_time_masks=2, num_freq_masks=2):
    """Apply simple SpecAugment on mel spectrogram (in-place copy)."""
    f, t = feat.shape
    aug = feat.copy()
    for _ in range(num_time_masks):
        t0 = random.randint(0, max(0, t - 1))
        w = random.randint(1, min(time_mask_param, t - t0))
        aug[:, t0:t0 + w] = 0
    for _ in range(num_freq_masks):
        f0 = random.randint(0, max(0, f - 1))
        w = random.randint(1, min(freq_mask_param, f - f0))
        aug[f0:f0 + w, :] = 0
    return aug


def augment_waveform(y, sr, p_noise=0.5, p_time=0.3, p_pitch=0.3):
    """Apply random augmentations to waveform with given probabilities."""
    if random.random() < p_noise:
        y = add_noise(y)
    if random.random() < p_time:
        y = random_time_stretch(y)
    if random.random() < p_pitch:
        y = random_pitch_shift(y, sr)
    return y
