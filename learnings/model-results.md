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
- **Test vectors and e2e clip:** `out/testvec/` holds 3 positive and 3 negative held-out clips with per-window expected outputs. `e2e_train.wav` is used for the live inject test. Since v0.5.0 it is the CC0 Wikimedia Commons recording "Railway crossing near Woodville" (trackside pass-by with horn), via `model/data/e2e_override.wav`. The earlier FSD50K eval/125125 turned out to be subway-interior speech; see iteration 3.

## Iteration 3: speech false positives and label cleaning (head v2, app v0.5.0)

**Symptom (live, on device).** Talking near the intercom triggered "Train passing".
- Speech windows averaged a train score of **0.60** (p90 0.85) while rail sounds were **0.00**.
- A 73 s event consisted entirely of speech.

**Root cause.** Many FSD50K rail-labelled clips are **subway interiors and station announcements**: speech with no audible train.
- The head learned "speech in this context = train".
- The old end-to-end test clip (eval/125125, "female speech, subway") was itself one of these: speech 0.83–0.99, rail 0.00, scored 0.6–0.97.

**Fix 1: label cleaning, retrained head v2.**
- Drop positive training windows with speech ≥ 0.5 and rail/horn < 0.1 (10,280 of 42,779 positive windows).
- Weight speech negatives ×3.

| Mean head score while speech ≥ 0.5 | v1 | v2 |
|---|---|---|
| Non-train clips (test) | 0.28 | **0.03** |
| Train-labelled clips (test) | 0.55 | **0.06** |

- Synthetic speech (Windows TTS, 6 clips, on device) now scores max **0.00–0.02**, down from as high as 0.53.
- Test clip-level P/R@0.5 is 0.78/0.65 (v1 was 0.72/0.72). The recall loss is mostly subway-interior speech clips no longer counting as trains, which is intended for trackside detection.

**Fix 2: sound rules (`fusion.c`).**
- Speech suppression is on by default: score × (1 − max(0, speech − rail evidence in the last ~3 s)).
- Optional rules: rail gate, loudness rise above the 60 s background, other horns near rail. Offline, a hard 6 dB rise gate halved recall, so it is off by default.

**Open-licensed web trains** (Wikimedia Commons, CC0/CC-BY/BY-SA, 15 pass-bys; none were in training), scored on device:
- YAMNet's rail lane peaks at **0.96–1.00** on nearly every real train, and the train-horn lane at **0.80–0.95** on clips with horns.
- With defaults (trigger 3 s), **10–11 of 15 fire**. With a 2 s trigger, 12 of 15, still with zero speech false positives.
- The misses are videos with narration or music over the train, or very short or distant passes.

**Horns.**
- Clean horn audio is recognized strongly.
- Steam and signal whistles land in "Other horns" (YAMNet *Steam whistle* = 1.00, *Train whistle* ≈ 0.2).
- Weak horns during **phone playback** point to the acoustic path: small speakers roll off below ~400 Hz, and horn fundamentals are ~250–600 Hz. The intercom's noise suppression and beamforming also attenuate sustained tonal sounds.
