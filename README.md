# Train Audio Detection (Axis ACAP)

An on-device ACAP that listens to an Axis device's microphone and raises an Axis event while a train is audibly passing. A train horn is flagged separately. Sensitivity, timing and level gates are adjustable with sliders.

> **Status:** proof of concept (v0.2.0). **Not a safety system.** Do not use it for level-crossing protection, worker warning or any life-safety function.

## How it works

1. **Capture.** A PipeWire stream captures the microphone (`Audio/Source`) as 16 kHz mono float into a ring buffer.
2. **Inference.** Every **0.48 s**, the newest **0.975 s** (15,600 samples) goes to **larod** (`cpu-tflite`). The model contains:
   - the log-mel front-end;
   - the **YAMNet** backbone (MobileNet v1, AudioSet-pretrained);
   - a small **trained train / not-train head** on YAMNet's 1024-d embedding.

   It outputs a train probability plus YAMNet's 521 class scores, which feed the horn detector and the "top sounds" display.
3. **Detection.** A state machine (`detector.c`) applies:
   - a loudness gate;
   - moving-average smoothing;
   - a threshold derived from **Sensitivity**;
   - a minimum **Trigger duration**;
   - a **Release** hang time that bridges gaps between cars;
   - a **Cooldown**.

   A horn sub-detector (YAMNet *Train horn* / *Train whistle*) can start an event on its own.
4. **Output.** Detections become a stateful **AXEvent** `CameraApplicationPlatform/TrainDetect/TrainDetected` with `active`, `horn` and `confidence`. You can use it in action rules (record, overlay, notify, I/O).

FastCGI endpoints (`status.cgi`, `timeline.cgi`, `analyze.cgi`, `label.cgi`) serve the settings UI and the test tools. They run behind the device's Apache, so device authentication and TLS apply.

The design rationale, datasets and prior work are in [docs/research.md](docs/research.md). Requirements are in [docs/requirements.md](docs/requirements.md).

## Prerequisites

- An AXIS OS **12.11.72+** device with a microphone, **aarch64** (ARTPEC-8/9). It was tested on an AXIS I8307-VE intercom (ARTPEC-8, AXIS OS 12.11.77).
- **Allow unsigned apps** enabled on the device (Apps page toggle). The package is not signed yet; see *Known limitations*.
- Build host: Docker (the official `axisecp/acap-native-sdk:12.11.0` image). On Windows, if Docker Desktop is unavailable, the Docker engine inside WSL works. See [learnings](learnings/device-and-build-notes.md).

## Build

```bash
# 1. model (CPU, ~1 h incl. downloads; see model/README.md)
cd model && docker build -t traindet-ml . && \
  docker run --rm -v "$PWD:/work" traindet-ml bash -c \
  "python -u scripts/01_download.py && python -u scripts/02_embed.py && python -u scripts/03_train_head.py && python -u scripts/04_export.py"
# 2. ACAP package -> build/Train_Audio_Detection_<version>_aarch64.eap
tools/build.sh aarch64
# 3. host unit tests (detector state machine + WAV parser, ASan/UBSan)
tools/test_host.sh
```

## Install and verify

Put `DEVICE_IP`, `DEVICE_USER` and `DEVICE_PASS` in `.env.devices`. That file is git-ignored and is used only by the dev scripts. Then:

```bash
python tools/device.py deploy            # uploads build/*aarch64.eap and starts it
python tools/device.py verify            # health, model parity vs host, live settings, event, end-to-end inject
python tools/device.py logs 40
```

You can also install by hand: device web UI › **Apps** › **Add app** › choose the `.eap`, then **Start**. Open the app's settings page to see the live view.

## Configuration

All settings apply live, with no restart. They can be changed from the app's settings page or with `param.cgi` (`root.Train_detect.*`).

| Setting | Default | Range | Effect |
|---|---|---|---|
| Enabled | yes | yes/no | Master switch for events. Inference and meters keep running. |
| Sensitivity | 50 | 1–100 | Threshold = 0.95 − 0.009 × (S − 1). At 50 the threshold is 0.51. Higher catches more trains and gives more false alarms. |
| TriggerSeconds | 3 | 0.5–30 s | Time the smoothed score must stay above the threshold before the event starts. |
| ReleaseSeconds | 8 | 1–120 s | Time below the threshold before the event ends. Bridges gaps. |
| SmoothingSeconds | 2 | 0.5–10 s | Moving-average window over the per-window score. |
| MinLevelDbfs | −65 | −90…−10 | Windows quieter than this score 0. |
| CooldownSeconds | 10 | 0–300 s | Minimum gap between events. |
| MaxEventSeconds | 600 | 0–3600 s | Force-ends an event after this long, so persistent noise above the threshold cannot latch it. 0 = no limit. |
| HornSensitivity | 50 | 1–100 | Horn threshold = 0.6 − 0.0055 × (S − 1). |
| HornTriggers | yes | yes/no | A detected horn starts an event immediately. |
| ScoreSource | head | head / yamnet / max | `head` is the trained detector. `yamnet` is YAMNet's rail classes, a baseline with a much higher false-alarm rate. `max` uses whichever of the two is higher. |

**To act on detections:** go to **System › Events › Rules**, choose the condition **Applications › Train Audio Detection › Train detected**, and pick any action.

### Site tuning

1. Watch the live score and level meters during real trains and during typical non-train noise (trucks, aircraft, wind).
2. Set **Minimum level** just above the quiet background.
3. Raise **Sensitivity** until trains reliably cross the threshold line.
4. Lengthen **Trigger duration** if short noises still trigger.
5. Use **Test with an audio file** on the settings page with recordings from the site. *Analyze* shows on-device scores; *Play into live detector* exercises the full event path.

## Operator UI

The settings page (Apps › Train Audio Detection › Open) is a single-page operator console with four views. On a phone it collapses to a bottom tab bar.

- **Live:** detection state, live score chart, sound level and horn meters, and the sounds the model recognizes right now.
- **Tuning:** site presets and every setting as a slider. A preview replays the last ~10 minutes of scores through the detector, so you see how many events your settings would have produced.
- **History:** stats, a 24-hour strip and a detections table with **"Was it a train?" feedback**, plus CSV export of events and labels.
- **Test audio:** drop any audio file, then *Analyze on device* (gives a verdict under the current settings) or *Play into live detector*.

The design source is the Claude Design canvas "Train Audio Detection — UI Redesign". To work on the UI without a device, run `python tools/ui_preview.py`, which serves `acap/app/html` with a mock API at http://localhost:8765.

## API

| Endpoint | Access | Description |
|---|---|---|
| `GET /local/train_detect/status.cgi` | operator | JSON: state, scores, level, key-sound groups, top-5 YAMNet classes, current event, history (last 50), inference and audio health, effective config |
| `GET /local/train_detect/timeline.cgi?since=<seq>` | operator | Per-window (0.48 s) rows newer than `seq`: head and smoothed score, level, state, and the key-sound groups (rail, train horn, other horns, road vehicles, speech, music). Covers the last ~5 minutes. |
| `POST /local/train_detect/analyze.cgi` | admin | Body: WAV, 16 kHz mono PCM16, ≤ 60 s. Returns per-window `prob`, `rail`, `horn`, `level_db`, key-sound `groups` and top-3 classes. |
| `POST /local/train_detect/analyze.cgi?mode=inject` | admin | Plays the WAV into the live detector in place of the mic, for commissioning and end-to-end tests. Resulting events carry `test=true`, both in the AXEvent and in history. |
| `POST /local/train_detect/label.cgi?start=<unix s>&label=train\|false_alarm\|none` | operator | Records operator feedback on a history event. Requires the header `X-Requested-With: traindet`. |

`analyze.cgi` requires `Content-Type: audio/wav` and rejects cross-origin `Origin` headers. Together with the custom header on `label.cgi`, this blocks cross-site request forgery from other pages open in an operator's browser. The app routes requests on the exact CGI name and rejects extra path segments, so one access level cannot reach another endpoint.

The AXEvent `TrainDetected` carries the data keys `active`, `horn`, `test` and `confidence`. To keep test playback out of production rules, add the condition `test = false`.

**Privacy:** audio is processed in memory and never stored or transmitted. Live sound classifications (`status.cgi`, `timeline.cgi`) are restricted to operator and admin, because they reveal what the microphone hears (for example, speech).

## Known limitations

- **First-swing model.** It was trained on Freesound and ESC-50 clips, not on trackside audio from your site or this mic. Expect domain shift; per-site tuning and site recordings are the next step (see `model/README.md`).
- **Data license.** The head was trained partly on **CC BY-NC** clips, so the bundled model is for **evaluation only**. Retrain on CC0/CC-BY clips plus your own recordings before any commercial use.
- **Unsigned.** `vendorId` is a dev placeholder (`0000000000`), so installing needs "Allow unsigned apps", a device-wide setting. Before any distribution, sign the package (the non-TIP ACAP Signing Service assigns the vendor ID) and drop that requirement.
- **Intercom audio DSP.** Noise cancelling, beamforming and echo cancelling are on by default and may attenuate rumble. Consider disabling noise cancelling on a dedicated train-detection device.
- **Limited sensing.** A 16 kHz mic loses wheel-squeal energy above 8 kHz. There is no direction, speed or train counting.
- **No persistence.** History and feedback labels are in memory only (last 50 events) and are lost on restart. Export the CSV regularly.
- **aarch64 only so far.** The armv7hf build is untested.

## License

MIT, see [LICENSE](LICENSE). Third-party components and the training-data terms are listed in [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md). YAMNet is Apache-2.0, with the full text in [LICENSE-APACHE-2.0](LICENSE-APACHE-2.0). The bundled model is for **evaluation and non-commercial use only**, because part of its training data is CC BY-NC.
