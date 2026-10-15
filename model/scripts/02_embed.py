"""Run YAMNet over every clip (plus augmented copies for the train split) and store
per-patch embeddings for head training.

Output: out/embeddings.npz with arrays
  emb (N,1024) float16, label (N,) int8, split (N,) int8 (0 train,1 val,2 test),
  clip (N,) int32 (row index in manifest), aug (N,) int8, rms_db (N,) float16,
  rail (N,) float16 (max YAMNet rail-class score), horn (N,) float16
"""
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "yamnet"))
import params as yamnet_params  # noqa: E402
import yamnet as yamnet_lib  # noqa: E402

from common import DATA, MANIFEST, OUT, SR, YAMNET_HORN, YAMNET_RAIL, read_wav16  # noqa: E402

PATCH = 15600
HOP = 7680
SPLITS = {"train": 0, "val": 1, "test": 2}
# Negative clips used as background beds when mixing positives for augmentation.
BED_LABELS = ("Wind", "Rain", "Traffic_noise_and_roadway_noise", "Speech", "Bird",
              "Car_passing_by", "Crowd", "Engine", "Thunderstorm", "Waves_and_surf")
AUGS_PER_TRAIN_CLIP = 2


def patch_rms_db(wave):
    """dBFS RMS of each analysis window, matching the device's windowing."""
    n = max(1, 1 + int(np.ceil(max(0, len(wave) - PATCH) / HOP)))
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        seg = wave[i * HOP:i * HOP + PATCH]
        out[i] = 10 * np.log10(np.mean(seg.astype(np.float64) ** 2) + 1e-12)
    return out


def lowpass(wave, rng):
    """One-pole low-pass with a random corner: crude air-absorption / distance."""
    fc = rng.uniform(800, 4000)
    a = np.exp(-2 * np.pi * fc / SR)
    from scipy.signal import lfilter
    return lfilter([1 - a], [1, -a], wave).astype(np.float32)


def augment(wave, label, beds, rng):
    w = wave.copy()
    if rng.random() < 0.5:
        w = lowpass(w, rng)
    if rng.random() < 0.3:  # small pitch/speed change (Doppler-ish), +-1.5 semitones
        factor = 2 ** (rng.uniform(-1.5, 1.5) / 12)
        idx = np.arange(0, len(w) - 1, factor)
        w = np.interp(idx, np.arange(len(w)), w).astype(np.float32)
    if beds and rng.random() < (0.7 if label else 0.4):
        bed = beds[rng.integers(len(beds))]
        if len(bed) < len(w):
            bed = np.tile(bed, int(np.ceil(len(w) / len(bed))))
        off = rng.integers(0, len(bed) - len(w) + 1)
        bed = bed[off:off + len(w)]
        p_sig = np.mean(w ** 2) + 1e-10
        p_bed = np.mean(bed ** 2) + 1e-10
        snr_db = rng.uniform(0, 20)
        w = w + bed * np.sqrt(p_sig / (p_bed * 10 ** (snr_db / 10)))
    gain_db = rng.uniform(-24, 6)
    w = w * 10 ** (gain_db / 20)
    peak = np.max(np.abs(w)) + 1e-9
    if peak > 1.0:
        w = w / peak
    return w.astype(np.float32)


def main():
    import tensorflow as tf
    params = yamnet_params.Params()
    model = yamnet_lib.yamnet_frames_model(params)
    model.load_weights(os.path.join(DATA, "yamnet.h5"))

    rows = list(csv.DictReader(open(MANIFEST, encoding="utf-8")))
    rng = np.random.default_rng(7)
    beds = []
    for r in rows:
        if r["split"] == "train" and r["label"] == "0" and any(b in r["labels"] for b in BED_LABELS):
            if len(beds) < 300:
                beds.append(read_wav16(os.path.join(DATA, r["path"])))
    print(f"{len(rows)} clips, {len(beds)} background beds", flush=True)

    out = {k: [] for k in ("emb", "label", "split", "clip", "aug", "rms_db", "rail", "horn")}

    @tf.function(input_signature=[tf.TensorSpec([None], tf.float32)])
    def run(w):
        s, e, _ = model(w)
        return s, e

    for ci, r in enumerate(rows):
        wave = read_wav16(os.path.join(DATA, r["path"]))
        if len(wave) < 1600:
            continue
        label = int(r["label"])
        versions = [(0, wave)]
        if r["split"] == "train":
            versions += [(a + 1, augment(wave, label, beds, rng)) for a in range(AUGS_PER_TRAIN_CLIP)]
        for aug, w in versions:
            scores, emb = run(tf.constant(w))
            scores, emb = scores.numpy(), emb.numpy()
            rms = patch_rms_db(w)
            n = min(len(emb), len(rms))
            out["emb"].append(emb[:n].astype(np.float16))
            out["rail"].append(scores[:n, YAMNET_RAIL].max(axis=1).astype(np.float16))
            out["horn"].append(scores[:n, YAMNET_HORN].max(axis=1).astype(np.float16))
            out["rms_db"].append(rms[:n].astype(np.float16))
            out["label"].append(np.full(n, label, np.int8))
            out["split"].append(np.full(n, SPLITS[r["split"]], np.int8))
            out["clip"].append(np.full(n, ci, np.int32))
            out["aug"].append(np.full(n, aug, np.int8))
        if ci % 250 == 0:
            print(f"  {ci}/{len(rows)} clips, {sum(len(x) for x in out['label'])} patches", flush=True)

    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(os.path.join(OUT, "embeddings.npz"),
                        **{k: np.concatenate(v) for k, v in out.items()})
    print("saved", sum(len(x) for x in out["label"]), "patches", flush=True)


if __name__ == "__main__":
    main()
