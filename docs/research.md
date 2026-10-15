# Research: Edge Audio Train Detection

_Compiled 2026-10-15. **[V]** = verified against the primary source (Zenodo API, GitHub raw, HF API, HEAD request, or the label CSVs). **[U]** = not verified. **[D]** = derived or engineering judgement._

## TL;DR

- **Approach.** Use a YAMNet backbone (MobileNet v1, pretrained on AudioSet) on 0.96 s log-mel patches. On top of its 1024-d embedding, add a small trained "train vs not-train" head. Then apply temporal logic: smoothing, hysteresis, minimum duration, a level gate and a hang time.
- **Why not use YAMNet's train classes directly?** A 2025 study of 16 trackside sites measured off-the-shelf YAMNet train classes at **6.1% frame precision** (84% recall). Trucks, motorcycles, aircraft and construction machinery fire them constantly. A trained head with hard negatives and duration logic is required. [V]
- **Data.** No public, commercially licensed trackside pass-by audio dataset exists. The first-swing model uses:
  - ESC-50: 40 train clips plus 1,960 diverse negatives.
  - FSD50K: about 836 rail-labelled clips plus confuser classes, fetched per-file from the HF mirror `Fhrozen/FSD50k`.

  Licenses are mixed (CC0 / CC-BY / CC-BY-NC), so the model is **POC / non-commercial** until it is retrained on CC0/CC-BY-only clips plus on-site recordings.
- **Biggest real-world lever.** Per-site calibration. Per-site thresholds raised event F1 from 70% to 84% in the 2025 study. The ACAP exposes sensitivity, duration and level gates as sliders for this reason.

## 1. Datasets

| Dataset | Train content | Format | Size | License | Commercial |
|---|---|---|---|---|---|
| **ESC-50** [V] | `train` (target 45), 40 × 5 s | 44.1 kHz mono WAV | ~600 MB | CC BY-NC 3.0. Per clip: 17 CC0 / 16 CC-BY / 7 BY-NC | Per clip |
| **FSD50K** [V] | Train 338+111, Rail_transport 637+199, Subway 305+82 (dev+eval) | 44.1 kHz WAV, 0.3–30 s | 24.7 GB full; per-file via HF `Fhrozen/FSD50k` | CC-BY dataset; clips CC0/BY/BY-NC/Sampling+ | Per clip; ask the authors |
| UrbanSound8K [V] | none (good hard negatives) | ≤4 s | 6.0 GB | CC BY-NC 4.0 | No |
| TAU/TUT Acoustic Scenes [V] | metro/tram/train *interior* scenes | 10–30 s | 8–36 GB | Non-commercial | No. These are cabin audio, not trackside. |
| AudioSet [V] | Rail transport 8.7k, Train 12.5k, Horn 2.2k (unbalanced) | YouTube IDs only; HF `agkphysics/AudioSet` parquet | huge | Labels CC-BY; audio belongs to YouTube | Unclear |
| DataSEC (Zenodo 23097290) [V] | `Train` plus truck/aircraft/thunder | 44.1 kHz WAV | 4.6 GB | CC BY-NC-SA | No |
| RATTLE (Zenodo 14780842) [V] | 2,116 real pass-bys | mel PNGs only | 173 MB | CC-BY | Yes, but not usable as audio input |

Two practical notes:

- **Per-file FSD50K.** `https://huggingface.co/datasets/Fhrozen/FSD50k/resolve/main/clips/{dev,eval}/<fname>.wav` (no auth) [V]. Labels come from `FSD50K.ground_truth.zip` on Zenodo record 4060432.
- **Missing labels.** FSD50K's curated vocabulary has **no** Train horn, Train whistle, Wheel squeal or Railroad car labels. Those appear only in the noisier pre-curation collection labels.

## 2. YAMNet [V]

- **Class indices** (`yamnet_class_map.csv`):
  - Rail classes: Rail transport 322, Train 323, Train whistle 324, Train horn 325, Railroad car 326, Train wheels squealing 327, Subway 328.
  - Main confusers: Vehicle 294, Truck 310, Air brake 311, Air horn/truck horn 312, Traffic noise 321, Aircraft 329, Jet engine 331, Engine 337, Heavy engine 343, Wind 277, Wind noise (mic) 279, Thunder 281, Rain 283, Rumble 487.
- **Features:**
  - 16 kHz, 25 ms periodic Hann window, 10 ms hop, 512-point FFT, magnitude spectrum.
  - 64 mel bands from 125 to 7500 Hz (HTK mel, `tf.signal.linear_to_mel_weight_matrix`), then `log(mel + 0.001)`.
  - Patches of 96 frames (0.96 s) with a 0.48 s hop.
- **Model size and accuracy:** 3.7 M weights, 69 M multiply-adds per patch. AudioSet mAP 0.306.
- **Sources:**
  - Weights: `https://storage.googleapis.com/audioset/yamnet.h5`
  - Code: `tensorflow/models/research/audioset/yamnet/{params,yamnet,features,export}.py`

## 3. Prior work

- **Betton-Ployon et al., JASA-EL 5(5) 053601, 2025.** "Robust automatic train pass-by detection combining deep learning and sound level analysis." [V]
  - Setup: YAMNet pre-filter, then LAeq envelope segmentation (Tmin 5 s, emergence ≥10 dB(A)), then a fine-tuned YAMNet.
  - Frame-level results: P 77 / R 90 / F1 83.
  - Event-level F1: 70% overall, **84% with per-site thresholds**.
  - By context: rural without a road 92%, urban 62%.
  - Main confusers: trucks, motorcycles, aircraft.
- **Bruitparif / Medusa, Forum Acusticum 2025.** [V]
  - Level thresholds alone (55–60 dB(A), 10 dB emergence, Tmin 5 s) over-detect: 511–876 events per day against 354 real trains.
  - Adding direction-of-arrival brought precision above 0.95.
- **Tran & Tsai, Sensors 2022 (THD-RawNet).** [V] Train-horn detector, 95% on clean audio and 83% at −15 dB SNR.
- **RATTLE, IEEE Access 2025.** [V] Edge mel-CNN, 96% train-type ID, model under 1.6 MB.
- **Commercial products.** No commercial off-the-shelf, audio-only train detector found. AXIS Audio Analytics has no vehicle or train classes.

## 4. Domain facts used for the detection logic

- **US horn rule, 49 CFR 222.21.** [V] Pattern long-long-short-long. Sounding starts 15–20 s (≤25 s) before the crossing. Level 96–110 dB(A) at 100 ft (49 CFR 229.129).
- **Event durations.**
  - AudioSet strong labels: horn median 1.6 s (p90 4.9 s); wheel squeal median 0.95 s.
  - Pass-bys: regional trains 8–10 s, intercity about 18 s, high-speed about 23 s (RATTLE, Bologna).
  - Freight trains 23–75+ s. [D]
- **Doppler.** At 25 m/s the pitch shifts about 2.5 semitones from approach to recede. [D] Augment with ±1.5 semitones.
- **Device limits.** The 16 kHz device mic plus YAMNet's 7.5 kHz mel ceiling lose the top of wheel-squeal energy (up to about 10–16 kHz). Rolling noise and engine energy fall well inside the band.

## 5. Recommended detection logic (implemented in the ACAP)

1. **Score.** Compute a per-patch probability every 0.48 s from the trained head.
2. **Smoothing.** Average over N seconds (default 2 s).
3. **Onset.** The smoothed score must stay ≥ threshold (set by sensitivity) for at least the *trigger duration* (default 3 s).
4. **Level gate.** Ignore patches quieter than the *minimum level* (dBFS). Optionally require emergence above the rolling background.
5. **Hang / release.** The event stays active until the score has been below threshold for the *release time* (default 8 s). This merges gaps.
6. **Cap.** Events longer than *max duration* are logged as "long" (idling or rumble) but kept active.
7. **Horn sub-detector.** YAMNet Train horn/whistle (324/325) ≥ horn threshold for ≥2 consecutive patches. It is reported in status and as an event field.
8. **Per-site tuning.** Adjust through the settings sliders, using the live score meter as a guide.

## Sources
- https://github.com/karolpiczak/ESC-50
- https://zenodo.org/records/4060432
- https://huggingface.co/datasets/Fhrozen/FSD50k
- https://zenodo.org/records/1203745
- https://research.google.com/audioset/download.html
- https://github.com/tensorflow/models/tree/master/research/audioset/yamnet
- https://pubs.aip.org/asa/jel/article/5/5/053601/3347187
- https://dael.euracoustics.org/confs/fa2025/data/articles/000561.pdf
- https://pmc.ncbi.nlm.nih.gov/articles/PMC9227093/
- https://zenodo.org/records/14780842
- https://www.law.cornell.edu/cfr/text/49/222.21
- https://www.law.cornell.edu/cfr/text/49/229.129
