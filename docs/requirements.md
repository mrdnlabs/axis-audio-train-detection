# Train Audio Detection ACAP: Requirements (POC v0.1)

**One sentence:** An on-device ACAP that listens to the device microphone and raises an Axis event while a train is audibly passing (with a horn flag), with operator-tunable sensitivity and timing.

## 1. Scope and suitability

| Item | Decision |
|---|---|
| Target | Any AXIS OS 12.x device with a microphone. aarch64 first (ARTPEC-8 / ARTPEC-9). |
| Test device | AXIS I8307-VE intercom, ARTPEC-8, AXIS OS 12.11.77, internal mic at 16 kHz |
| Inference | Larod `cpu-tflite` (float32). Optional ARTPEC-8/9 DLPU later, which needs int8 per-tensor QAT. |
| Safety | **Not a safety system.** It must not be used for level-crossing protection, worker warning, or any life-safety function. It is an informational and analytics signal only (standards §2). |
| Licensing | App code is MIT. YAMNet code and weights are Apache-2.0. The first-swing head is trained on mixed CC0, CC-BY and CC-BY-NC clips, so it is for **POC / evaluation only**. Retrain on CC0/CC-BY plus own recordings before any commercial use. |

## 2. Functional requirements

| ID | Requirement |
|---|---|
| F1 | Capture mono audio from the device's microphone source through PipeWire. Convert to 16 kHz float. |
| F2 | Every 0.48 s, run the model on the latest 0.975 s window (15,600 samples). Outputs are train probability (trained head) and 521 YAMNet class scores. |
| F3 | Compute the window level in dBFS (RMS) for the loudness gate and for display. |
| F4 | Detection state machine: IDLE → CANDIDATE → ACTIVE → (release) → IDLE, using smoothed score, threshold, trigger duration, release time and level gate. |
| F5 | Horn sub-detector: max(YAMNet Train horn, Train whistle) ≥ horn threshold for ≥2 consecutive windows. Sets the `horn` flag on the active event and in status. |
| F6 | Declare a stateful **AXEvent** "Train detected" (`Active` true/false, `Confidence`, `Horn`) usable in Axis action rules (record, overlay, notify, output port). |
| F7 | Keep an in-memory history of the last 50 detections: start, end, duration, peak and mean confidence, peak level, horn. |
| F8 | `status.cgi` (operator), plus `timeline.cgi` (operator, per-window key-sound history for about 5 minutes): live state, scores, level, top YAMNet classes, inference latency, counters, history, effective settings. |
| F9 | `analyze.cgi` (admin, POST WAV 16 kHz mono PCM16, ≤ 60 s): run the same model on uploaded audio and return per-window scores. Used for on-device verification and site tuning. `?mode=inject` plays the WAV into the live detector in place of the mic, for end-to-end event tests. Requires `Content-Type: audio/wav` and a same-origin request (CSRF guard). |
| F10 | Settings page with sliders. All settings take effect live, with no restart. |
| F11 | Structured syslog lines (`[traindet] ...`) for state transitions and errors. Rate-limited. |

## 3. Settings (AXParameter, all live)

| Param | UI | Range | Default | Meaning |
|---|---|---|---|---|
| `Enabled` | toggle | yes/no | yes | Master switch. Inference keeps running for the meter, but events are suppressed. |
| `Sensitivity` | slider | 1–100 | 50 | Maps to the detection threshold, `thr = 0.95 − 0.009 × (S−1)`. 1 → 0.95 (strict), 50 → 0.51, 100 → 0.06 (loose). |
| `TriggerSeconds` | slider | 0.5–30 s | 3 | How long the smoothed score must stay above threshold before the event fires. |
| `ReleaseSeconds` | slider | 1–120 s | 8 | Hang time. The event ends after the score has been below threshold this long. Merges gaps. |
| `SmoothingSeconds` | slider | 0.5–10 s | 2 | Moving-average window over per-window scores. |
| `MinLevelDbfs` | slider | −90 to −10 | −65 | Windows quieter than this are scored 0 (loudness gate). |
| `HornSensitivity` | slider | 1–100 | 50 | Horn threshold, `0.6 − 0.0055 × (S−1)`. |
| `CooldownSeconds` | slider | 0–300 s | 10 | Minimum gap after an event ends before a new one can start. |
| `MaxEventSeconds` | slider | 0–3600 s | 600 | Force-ends an event after this long (0 = no limit). |
| `HornTriggers` | toggle | yes/no | yes | A detected train horn starts an event immediately. Horns precede the train by 15–25 s at US crossings. |
| `ScoreSource` | select | head / yamnet / max | head | `head` = trained head. `yamnet` = max of YAMNet rail classes (baseline). `max` = the larger of the two. |

## 4. Non-functional

| ID | Requirement |
|---|---|
| N1 | CPU: one inference per 0.48 s. Target <150 ms per inference on ARTPEC-8 (≈30% of one core worst case). |
| N2 | Memory under 100 MB RSS. Model ≤ 20 MB. |
| N3 | Runs as `respawn` and starts at boot. Recovery:<br>• After about 10 s of consecutive inference failures, or 60 s without audio that had been streaming, it ends any active event (`active=false` is sent) and exits so it is respawned.<br>• If no audio has arrived since start, it reports that in status instead of crash-looping.<br>• Startup failures back off for 30 s before exiting. |
| N4 | Security: endpoints only via Apache FastCGI with device auth. Settings and analyze require admin; status requires viewer. No new ports, no stored credentials, input validated (size, format), output JSON-escaped. |
| N5 | Detection latency: event fires within TriggerSeconds + SmoothingSeconds/2 + 0.5 s of the train becoming dominant. |

## 5. Verification plan

1. **Offline.** Measure head accuracy on held-out ESC-50 fold 5 plus the FSD50K eval split: patch AUC, clip-level precision/recall and confusion against confusers. Check TFLite parity against Keras (max abs diff < 1e-3).
2. **On-device model parity.** POST test WAVs (train and non-train) to `analyze.cgi` and compare per-window scores with the host TFLite output.
3. **Live pipeline.** `status.cgi` shows a moving level and increasing inference counter, latency is within budget, and syslog is clean.
4. **Event path.** Feed train audio, then confirm the state goes ACTIVE, the event appears in the event list (`GetEventInstances`) and in history.
5. **Settings.** Change a slider through the UI or `param.cgi`, then confirm `status.cgi` reflects the effective value without a restart.
6. **Field test (user).** Real trains plus a negative day (trucks, aircraft, weather). Tune sensitivity per site.

## 6. Known limitations (first swing)

- Training data is mostly Freesound recordings, not trackside audio from this site or this device's mic. Expect domain shift; per-site calibration and own recordings are the next step.
- The intercom's DSP (noise cancelling, beamforming, echo cancelling) is enabled. It may suppress stationary rumble. Consider disabling `NoiseCancelling` for this use case, but that is the operator's call.
- The 16 kHz mic loses wheel-squeal energy above 8 kHz.
- No direction or speed estimation. No train counting beyond one event per pass-by.
