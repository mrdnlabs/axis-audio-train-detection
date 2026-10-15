# Train detection model

The device model is a single TFLite graph:

```
waveform (1, 15600) float32 @16 kHz
  └─ log-mel front-end (25 ms Hann / 10 ms hop, 512-pt DFT-by-matmul, 64 mel 125–7500 Hz, log(x+0.001))
     └─ YAMNet backbone (MobileNet v1, AudioSet-pretrained, Apache-2.0)
        ├─ 521 AudioSet class scores ─┬─> output "b_yamnet_scores" (horn detector, top-sounds UI, baseline)
        └─ 1024-d embedding ──────────┴─> concat (1545) -> trained head
                                          (Dropout, Dense 128 relu, Dropout, Dense 1 sigmoid, bias-calibrated)
                                          -> output "a_train_prob"
```

The front-end runs inside the graph, so the device only feeds raw audio. There is no feature code in C that could drift from training.

## Pipeline

Run everything in the `traindet-ml` container (CPU-only TensorFlow 2.17 + tf_keras):

| Step | Script | Output |
|---|---|---|
| 1 | `01_download.py`: ESC-50 (all 2,000 clips) + FSD50K: every rail-labelled clip (836) + capped confuser classes + random negatives, via the per-file HF mirror. Resampled to 16 kHz PCM16. | `data/clips/`, `data/manifest.csv` (incl. per-clip license) |
| 2 | `02_embed.py`: YAMNet embeddings per 0.96 s window (0.48 s hop). The train split also gets 2 augmented copies: low-pass "distance", ±1.5-semitone "Doppler", mixing with wind/rain/traffic/speech beds at 0–20 dB SNR, −24…+6 dB gain. | `out/embeddings.npz` |
| 3 | `03_train_head.py`: trains the head. Weak clip labels are cleaned by dropping windows < −70 dBFS and positive-clip windows > 25 dB below the clip peak. | `out/head.h5`, `out/metrics.json` |
| 4 | `04_export.py`: builds the waveform→[prob, scores] graph, converts to TFLite (builtin ops only), checks Keras↔TFLite parity, and writes test vectors for on-device parity checks. | `out/traindet.tflite`, `out/model_info.json`, `out/testvec/` |

```bash
docker build -t traindet-ml .
docker run --rm -v "$PWD:/work" traindet-ml bash -c "python -u scripts/01_download.py && python -u scripts/02_embed.py && python -u scripts/03_train_head.py && python -u scripts/04_export.py"
```

Splits:
- ESC-50: folds 1–3 train, 4 val, 5 test.
- FSD50K: dev `train`/`val` split for train/val, eval for test. There is no clip overlap between splits.

## Results (first swing)

See `out/metrics.json` after training. The headline numbers are recorded in [../learnings/model-results.md](../learnings/model-results.md).

## Next steps for a better model

1. **Site data.** Record 1–2 weeks at the target site with this device's mic, then label pass-bys. Use the app's history plus a short review of the recordings. This is the single biggest improvement; the 2025 literature reports +14 F1 from per-site calibration alone.
2. **Commercial-clean retrain.** Filter `manifest.csv` to CC0/CC-BY licenses and add site data.
3. **More rail data.** Use AudioSet strong labels via the `agkphysics/AudioSet` HF parquet mirror. It has ~12k Train and ~2k Train horn clips.
4. **DLPU.** Quantization-aware training with per-tensor int8 for the ARTPEC-8/9 DLPU, to cut CPU load. The CPU path is fine for a single model at 2 Hz.
5. **Fine-tune upper YAMNet layers** once there is enough in-domain data.
