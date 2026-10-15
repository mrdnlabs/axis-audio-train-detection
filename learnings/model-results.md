# Model results: first swing (2026-10-15)

## Data
- **7,161 clips** at 16 kHz:
  - ESC-50: 2,000 clips, 40 train.
  - FSD50K: 5,161 clips. That covers all 836 rail-labelled clips, capped confuser classes and random negatives.
- **Splits (no overlap):**
  - train: 4,576 clips (565 positive)
  - val: 906 (104)
  - test: 1,679 (207)
- **Windows:** 248k train (including 2 augmented copies per clip), 17k val and 38k test, each 0.96 s.

## Iteration 1: head on the embedding only, positives up-weighted
Results on held-out test clips at threshold 0.5:
- Clip AUC 0.938 / AP 0.783, but P 0.42 / R 0.90.
- Very high confuser false-positive rates: Bus 70%, Truck 50%, Aircraft 51%, Engine 36%.
- Lesson: the class-weighting made the head over-eager, and the embedding alone did not separate road vehicles well enough with this little data.

## Iteration 2 (shipped): embedding + YAMNet's 521 class scores, hard negatives ×2, bias-calibrated
Calibration moves the validation clip-level F1-optimal threshold (0.49) to 0.5, which is Sensitivity ≈ 50 on the device.

| Held-out test (1,679 clips) | Head (shipped) | YAMNet rail classes only |
|---|---|---|
| Clip ROC-AUC | **0.947** | 0.907 |
| Clip average precision | **0.783** | 0.732 |
| Clip P / R / F1 @0.5 (2 s smoothing) | 0.72 / 0.72 / **0.72** | 0.90 / 0.44 / 0.60 |
| Event rule @0.5, 1.5 s trigger | 0.77 / 0.64 / **0.70** | — |
| Event rule @0.5, 3 s trigger | 0.84 / 0.47 / 0.60 | 0.94 / 0.22 / 0.35 |

Confuser false-positive rate at 0.5 (clip level, test):

| Class | Rate |
|---|---|
| Truck | 7% |
| Bus | 19% |
| Aircraft | 16% |
| Fixed-wing aircraft | 22% |
| Engine | 9% |
| Car passing by | 4% |
| Traffic | 5% |
| Motorcycle | 2% |
| Thunder | 0% |
| Wind (FSD) | 0% |
| Siren | 0% |

**Aircraft are the main remaining confuser**, consistent with Betton-Ployon et al. 2025.

Caveats:
- **Event recall with a 3 s trigger is understated.** Many FSD50K "train" clips are only 1–5 s long, so they cannot satisfy a 3 s rule. Real pass-bys last 8–75 s.
- **On-device scores need not match these numbers.** The test data is Freesound audio, not this intercom's mic or DSP chain; check with `analyze.cgi` and site recordings.

## Export
- **Model file:** TFLite float32, **16.9 MB**.
- **Ops:** builtin only (ADD, CONCATENATION, CONV_2D, DEPTHWISE_CONV_2D, FULLY_CONNECTED, GATHER, LOG, LOGISTIC, MEAN, MUL, PAD, RESHAPE, SQRT).
- **Parity:** Keras vs TFLite max abs diff 3.8e-6.
- **Interface:** input `[1, 15600]` float32; outputs `[1,1]` (train prob) and `[1,521]` (YAMNet scores). The device tells the outputs apart by byte size.
- **Test vectors and e2e clip:** `out/testvec/` holds 3 positive and 3 negative held-out clips with per-window expected outputs. `e2e_train.wav` (FSD50K eval/125125, 28.7 s, 31 windows above 0.6 smoothed) is used for the live inject test.
