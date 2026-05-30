
import librosa
import librosa.effects as lfx
import numpy as np
import os
import tqdm
import glob
import random

# ============================================================
#  CONFIGURACIÓN GLOBAL DE AUGMENTATION
#  Cambia estos valores para controlar cuántas versiones
#  adicionales se generan por cada audio original.
# ============================================================
AUG_MULTIPLIER_PD      = 4   # Cada audio PD genera N versiones extra (recomendado: 3-5)
AUG_MULTIPLIER_HEALTHY = 4   # Igualado a PD para equilibrar clases (~2000 sanos vs ~2000 PD)
RANDOM_SEED            = 42


# ============================================================
#  TÉCNICAS DE AUGMENTATION
#  Cada función recibe (y, sr) y devuelve un array numpy.
#  Están calibradas para voz patológica — rangos conservadores
#  para no destruir los marcadores acústicos del Parkinson.
# ============================================================

def aug_pitch_shift(y, sr):
    """
    Desplaza el tono ±1-2 semitonos.
    Simula variabilidad natural entre grabaciones del mismo paciente.
    Rango conservador: ±2 st máximo para preservar el temblor vocal.
    """
    n_steps = random.uniform(-2.0, 2.0)
    return lfx.pitch_shift(y, sr=sr, n_steps=n_steps)


def aug_time_stretch(y, sr):
    """
    Estira o comprime el tiempo (0.85x – 1.15x).
    Simula diferencias en la velocidad del habla entre sesiones.
    No altera el tono, solo la duración.
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
    Añade ruido blanco gaussiano muy suave (SNR ~30 dB).
    Simula diferencias en el micrófono o el entorno.
    """
    noise_amp = 0.003 * np.max(np.abs(y))
    noise = noise_amp * np.random.randn(len(y))
    return y + noise


def aug_room_simulation(y, sr):
    """
    Convuelve con una respuesta de sala sintética (reverb simple).
    Simula que el paciente graba en habitaciones distintas.
    Usa un filtro exponencial decreciente como IR aproximada.
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
    Escala el volumen entre 0.7x y 1.3x.
    Simula que el paciente habla más cerca o lejos del micro.
    """
    factor = random.uniform(0.7, 1.3)
    return np.clip(y * factor, -1.0, 1.0)


def aug_combined(y, sr):
    """
    Combina noise + pitch shift suave.
    Produce la variante más 'diferente' al original
    sin perder las características patológicas.
    """
    y = aug_add_noise(y, sr)
    n_steps = random.uniform(-1.0, 1.0)   # rango más suave que pitch_shift sola
    y = lfx.pitch_shift(y, sr=sr, n_steps=n_steps)
    return y


# Lista ordenada de técnicas. El pipeline rota por ellas en orden
# para que cada muestra reciba un tipo de augmentation distinto.
AUGMENTATION_TECHNIQUES = [
    aug_pitch_shift,
    aug_time_stretch,
    aug_add_noise,
    aug_room_simulation,
    aug_volume_shift,
    aug_combined,
]


# ============================================================
#  EXTRACCIÓN DE FEATURES
# ============================================================

def _compute_mfcc_features(y, sr, n_mfcc=40):
    """
    Extrae MFCC + delta + delta² y devuelve tensor (frames, n_mfcc, 3).
    """
    mfcc       = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    delta_mfcc = librosa.feature.delta(mfcc)
    delta2     = librosa.feature.delta(mfcc, order=2)
    features   = np.stack([mfcc, delta_mfcc, delta2], axis=-1)
    return np.transpose(features, (1, 0, 2))   # (frames, 40, 3)


def extract_features_crnn(file_path, duration=3, n_mfcc=40):
    """
    Lee un archivo de audio y devuelve sus features sin augmentation.
    Devuelve None si el archivo no se puede leer.
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
    Lee un archivo y genera (1 original + n_augmentations variantes).

    Parámetros
    ----------
    file_path       : ruta al audio
    n_augmentations : cuántas versiones extra crear (recomendado 3-5)
    duration        : segundos de audio a usar
    n_mfcc          : número de coeficientes MFCC

    Devuelve
    --------
    list[np.ndarray] — lista de tensores (frames, 40, 3), puede estar
                       vacía si el archivo no se puede leer.
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

    # --- Original ---
    feat = _compute_mfcc_features(y, sr, n_mfcc)
    results.append(feat)

    # --- Versiones augmentadas ---
    for i in range(n_augmentations):
        technique = AUGMENTATION_TECHNIQUES[i % len(AUGMENTATION_TECHNIQUES)]
        try:
            y_aug = technique(y, sr)
            # Garantizar longitud correcta tras la transformación
            if len(y_aug) < target:
                y_aug = np.pad(y_aug, (0, target - len(y_aug)))
            elif len(y_aug) > target:
                y_aug = y_aug[:target]
            feat_aug = _compute_mfcc_features(y_aug, sr, n_mfcc)
            results.append(feat_aug)
        except Exception:
            # Si una técnica falla (ej. librosa edge case), la saltamos
            continue

    return results


# ============================================================
#  CARGA DE DATASETS CON AUGMENTATION
# ============================================================

def load_datasets(path_kcl_root, path_librispeech, path_db_it,
                  max_samples=1000,
                  aug_pd=AUG_MULTIPLIER_PD,
                  aug_healthy=AUG_MULTIPLIER_HEALTHY):
    """
    Carga y preprocesa los tres datasets con augmentation configurable.

    Parámetros
    ----------
    aug_pd      : versiones extra por audio PD      (default: 4)
    aug_healthy : versiones extra por audio sano    (default: 1)

    Con aug_pd=4 y 400 audios PD originales obtendrás ~2000 muestras PD.
    """
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    X, y = [], []

    # --- Buscar archivos ---
    # Solo carpeta ReadText — SpontaneousDialogue se excluye porque
    # el audio grabado contiene la voz del entrevistador, no del paciente.
    print(f"[1/4] Buscando casos Parkinson (ReadText) en {path_kcl_root}...")
    files_pd = glob.glob(
        os.path.join(path_kcl_root, "**/ReadText/PD/**/*.wav"), recursive=True
    )

    print(f"[2/4] Buscando sanos (ReadText) en {path_kcl_root}...")
    files_hc_kcl = glob.glob(
        os.path.join(path_kcl_root, "**/ReadText/HC/**/*.wav"), recursive=True
    )

    print(f"[3/4] Buscando LibriSpeech en {path_librispeech}...")
    files_ls = glob.glob(os.path.join(path_librispeech, "**/*.flac"), recursive=True)
    files_healthy = files_hc_kcl + files_ls

    print(f"[4/4] Buscando DB_IT en {path_db_it}...")
    files_db_it = glob.glob(os.path.join(path_db_it, "**", "*.wav"), recursive=True)
    files_pd = files_pd + files_db_it

    # --- Mezclar y recortar ---
    random.shuffle(files_pd)
    random.shuffle(files_healthy)
    files_pd      = files_pd[:max_samples]
    files_healthy = files_healthy[:max_samples]

    print(f"\nArchivos PD originales a procesar:    {len(files_pd)}")
    print(f"Archivos sanos originales a procesar: {len(files_healthy)}")
    print(f"Augmentation PD:      x{aug_pd + 1}  → ~{len(files_pd) * (aug_pd + 1)} muestras PD")
    print(f"Augmentation sanos:   x{aug_healthy + 1}  → ~{len(files_healthy) * (aug_healthy + 1)} muestras sanas")
    print()

    # --- Procesar PD con augmentation ---
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

    print(f"  PD procesados: {pd_ok} archivos × {aug_pd+1} = {pd_ok*(aug_pd+1)} muestras  |  {pd_skip} fallidos")

    # --- Procesar sanos con augmentation (más ligero) ---
    hc_ok = hc_skip = 0
    for f in tqdm.tqdm(files_healthy, desc="Procesando sanos + augmentation"):
        variants = extract_features_with_augmentation(f, n_augmentations=aug_healthy)
        if not variants:
            hc_skip += 1
            continue
        for feat in variants:
            X.append(feat)
            y.append(0)
        hc_ok += 1

    print(f"  HC procesados: {hc_ok} archivos × {aug_healthy+1} = {hc_ok*(aug_healthy+1)} muestras  |  {hc_skip} fallidos")

    X_arr = np.array(X)
    y_arr = np.array(y)

    print(f"\nDataset final: {X_arr.shape[0]} muestras totales")
    print(f"  PD (1): {np.sum(y_arr == 1)}  |  Sano (0): {np.sum(y_arr == 0)}")

    return X_arr, y_arr
