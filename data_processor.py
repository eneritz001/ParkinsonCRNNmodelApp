
import librosa
import librosa.effects as lfx
import numpy as np
import os
import tqdm
import glob
import random

#  GLOBAL SETTINGS FOR AUGMENTATION
AUG_MULTIPLIER_PD      = 4   # Each PD audio file generates N additional versions
AUG_MULTIPLIER_HEALTHY = 4   # Adjusted to PD to balance classes
RANDOM_SEED            = 42 



#  AUGMENTATION TECHNIQUES
# functions they are calibrated for pathological speech conservative thresholds

def aug_pitch_shift(y, sr):
    """
    Shifts the pitch by ±1–2 semitones.
    Simulates natural variation between recordings from the same patient.
    Conservative range: ±2 semitones maximum to preserve vocal tremor.
    """
    n_steps = random.uniform(-2.0, 2.0)
    return lfx.pitch_shift(y, sr=sr, n_steps=n_steps)


def aug_time_stretch(y, sr):
    """
    Stretches or compresses the duration (0.85x – 1.15x).
    Simulates variations in speaking rate between sessions.
    Does not alter the pitch, only the duration.
    """
    rate = random.uniform(0.85, 1.15)
    y_stretched = lfx.time_stretch(y, rate=rate)
    # Reajustar a la longitud original para mantener el tensor uniforme
    target = len(y)
    if len(y_stretched) > target:
        return y_stretched[:target]
    return np.pad(y_stretched, (0, target - len(y_stretched)))


def aug_add_noise(y, sr):
    """
    Adds very gentle Gaussian white noise (SNR ~30 dB).
    Simulates variations in the microphone or the environment.
    """
    noise_amp = 0.003 * np.max(np.abs(y))
    noise = noise_amp * np.random.randn(len(y))
    return y + noise


def aug_room_simulation(y, sr):
    """
    Involves a synthetic room response (simple reverb).
    Simulates the patient recording in different rooms.
    Uses a decaying exponential filter as an approximate IR.
    """
    decay = random.uniform(0.3, 0.6)
    ir_length = int(sr * 0.2)           # IR de 200 ms
    ir = np.exp(-decay * np.linspace(0, 5, ir_length))
    ir /= np.sum(ir)
    y_conv = np.convolve(y, ir, mode='full')[:len(y)]
    # Normalizar para no cambiar el volumen
    if np.max(np.abs(y_conv)) > 0:
        y_conv = y_conv / np.max(np.abs(y_conv)) * np.max(np.abs(y))
    return y_conv


def aug_volume_shift(y, sr):
    """
    Adjust the volume between 0.7x and 1.3x.
    Simulates the patient speaking closer to or further away from the microphone.
    """
    factor = random.uniform(0.7, 1.3)
    return np.clip(y * factor, -1.0, 1.0)


def aug_combined(y, sr):
    """
    Combines noise with a subtle pitch shift.
    Produces the version that differs most from the original
    without losing its distinctive characteristics.
    """
    y = aug_add_noise(y, sr)
    n_steps = random.uniform(-1.0, 1.0)   # rango más suave que pitch_shift sola
    y = lfx.pitch_shift(y, sr=sr, n_steps=n_steps)
    return y


# Ordered list of techniques. The pipeline cycles through them in order
# so that each sample receives a different type of augmentation.
AUGMENTATION_TECHNIQUES = [
    aug_pitch_shift,
    aug_time_stretch,
    aug_add_noise,
    aug_room_simulation,
    aug_volume_shift,
    aug_combined,
]


#  FEATURE EXTRACTION

def _compute_mfcc_features(y, sr, n_mfcc=40):
    """
    Extracts MFCCs, delta and delta², and returns a tensor (frames, n_mfcc, 3).
    """
    mfcc       = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    delta_mfcc = librosa.feature.delta(mfcc)
    delta2     = librosa.feature.delta(mfcc, order=2)
    features   = np.stack([mfcc, delta_mfcc, delta2], axis=-1)
    return np.transpose(features, (1, 0, 2))   # (frames, 40, 3)


def extract_features_crnn(file_path, duration=3, n_mfcc=40):
    """
    Reads an audio file and returns its features without augmentation.
    Returns None if the file cannot be read.
    """
    try:
        y, sr = librosa.load(file_path, sr=22050, duration=duration)
        target = sr * duration
        if len(y) < target:
            y = np.pad(y, (0, target - len(y)))
        return _compute_mfcc_features(y, sr, n_mfcc)
    except Exception:
        return None


def extract_features_with_augmentation(file_path, n_augmentations=4,
                                        duration=3, n_mfcc=40):
    """
    Reads a file and generates (1 original + n_augmentations variants).

    Parameters
    ----------
    file_path       : path to the audio file
    n_augmentations : number of additional versions to create (recommended 3–5)
    duration        : number of seconds of audio to use
    n_mfcc          : number of MFCC coefficients

    Returns
    --------
    list[np.ndarray] — list of tensors (frames, 40, 3); may be
                       empty if the file cannot be read.
    """
    try:
        y, sr = librosa.load(file_path, sr=22050, duration=duration)
        target = sr * duration
        if len(y) < target:
            y = np.pad(y, (0, target - len(y)))
        elif len(y) > target:
            y = y[:target]
    except Exception:
        return []

    results = []

    # Original
    feat = _compute_mfcc_features(y, sr, n_mfcc)
    results.append(feat)

    # Augmented versions
    for i in range(n_augmentations):
        technique = AUGMENTATION_TECHNIQUES[i % len(AUGMENTATION_TECHNIQUES)]
        try:
            y_aug = technique(y, sr)
            if len(y_aug) < target:
                y_aug = np.pad(y_aug, (0, target - len(y_aug)))
            elif len(y_aug) > target:
                y_aug = y_aug[:target]
            feat_aug = _compute_mfcc_features(y_aug, sr, n_mfcc)
            results.append(feat_aug)
        except Exception:
            continue

    return results


#  LOADING DATASETS WITH AUGMENTATION

def load_datasets(path_kcl_root, path_librispeech, path_db_it,
                  max_samples=1000,
                  aug_pd=AUG_MULTIPLIER_PD,
                  aug_healthy=AUG_MULTIPLIER_HEALTHY):
    """
    Loads and pre-processes the three datasets using configurable augmentation.

    Parameters
    ----------
    aug_pd      : extra versions per PD audio      (default: 4)
    aug_healthy : extra versions per healthy audio    (default: 1)

    With aug_pd=4 and 400 original PD audio files, you will obtain ~2000 PD samples.
    """
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    X, y = [], []

    # Search for files
    # Only the ReadText — SpontaneousDialogue folder is excluded
    print(f"[1/4] Searching for Parkinson's cases (ReadText) in {path_kcl_root}...")
    files_pd = glob.glob(
        os.path.join(path_kcl_root, "**/ReadText/PD/**/*.wav"), recursive=True
    )

    print(f"[2/4] Searching for healthy cases (ReadText) in {path_kcl_root}...")
    files_hc_kcl = glob.glob(
        os.path.join(path_kcl_root, "**/ReadText/HC/**/*.wav"), recursive=True
    )

    print(f"[3/4] Searching LibriSpeech in {path_librispeech}...")
    files_ls = glob.glob(os.path.join(path_librispeech, "**/*.flac"), recursive=True)
    files_healthy = files_hc_kcl + files_ls

    print(f"[4/4] Searching DB_IT in {path_db_it}...")
    files_db_it = glob.glob(os.path.join(path_db_it, "**", "*.wav"), recursive=True)
    files_pd = files_pd + files_db_it

    # Mix and cut
    random.shuffle(files_pd)
    random.shuffle(files_healthy)
    files_pd      = files_pd[:max_samples]
    files_healthy = files_healthy[:max_samples]

    print(f"\nOriginal PD files to be processed:    {len(files_pd)}")
    print(f"Original healthy files to be processed: {len(files_healthy)}")
    print(f"Augmentation PD:      x{aug_pd + 1}  → ~{len(files_pd) * (aug_pd + 1)} muestras PD")
    print(f"Augmentation healthy:   x{aug_healthy + 1}  → ~{len(files_healthy) * (aug_healthy + 1)} muestras sanas")
    print()

    # Process PD with augmentation 
    pd_ok = pd_skip = 0
    for f in tqdm.tqdm(files_pd, desc="Procesando PD + augmentation"):
        variants = extract_features_with_augmentation(f, n_augmentations=aug_pd)
        if not variants:
            pd_skip += 1
            continue
        for feat in variants:
            X.append(feat)
            y.append(1)
        pd_ok += 1

    print(f"  PD processed: {pd_ok} files × {aug_pd+1} = {pd_ok*(aug_pd+1)} samples  |  {pd_skip} unsuccessful")

    # Process healthy files with augmentation
    hc_ok = hc_skip = 0
    for f in tqdm.tqdm(files_healthy, desc="Processing healthy + augmentation"):
        variants = extract_features_with_augmentation(f, n_augmentations=aug_healthy)
        if not variants:
            hc_skip += 1
            continue
        for feat in variants:
            X.append(feat)
            y.append(0)
        hc_ok += 1

    print(f"   HC processed: {hc_ok} files × {aug_healthy+1} = {hc_ok*(aug_healthy+1)} samples  |  {hc_skip} unsuccessful")

    X_arr = np.array(X)
    y_arr = np.array(y)

    print(f"\nFinal dataset: {X_arr.shape[0]} total samples")
    print(f"  PD (1): {np.sum(y_arr == 1)}  |  Healthy (0): {np.sum(y_arr == 0)}")

    return X_arr, y_arr
