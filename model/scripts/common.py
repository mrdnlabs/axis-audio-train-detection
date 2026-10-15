"""Shared helpers for the train-detection model pipeline."""
import io
import os

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(ROOT, "data")
CLIPS = os.path.join(DATA, "clips")
OUT = os.path.join(ROOT, "out")
MANIFEST = os.path.join(DATA, "manifest.csv")
SR = 16000

# YAMNet class indices (yamnet_class_map.csv)
YAMNET_RAIL = [322, 323, 324, 325, 326, 327, 328]
YAMNET_HORN = [324, 325]


def to_16k_mono(data, sr):
    """Float32 mono at 16 kHz from any soundfile-decoded array."""
    data = np.asarray(data, dtype=np.float32)
    if data.ndim == 2:
        data = data.mean(axis=1)
    if sr != SR:
        g = np.gcd(int(sr), SR)
        data = resample_poly(data, SR // g, int(sr) // g).astype(np.float32)
    return np.clip(data, -1.0, 1.0)


def decode_bytes(raw):
    data, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
    return to_16k_mono(data, sr)


def write_wav16(path, wave):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, wave, SR, subtype="PCM_16")


def read_wav16(path):
    data, sr = sf.read(path, dtype="float32", always_2d=False)
    return to_16k_mono(data, sr)
