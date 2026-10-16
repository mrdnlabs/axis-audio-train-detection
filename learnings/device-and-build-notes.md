# Learnings: device, build environment, platform (2026-10-15)

## Test device
- **AXIS I8307-VE Network Video Intercom**: ARTPEC-8, AXIS OS **12.11.77**, aarch64. Embedded development v2.18.
- **Audio source A0.** Internal mic at **16 kHz** mono (`root.AudioSource.A0.SampleRate=16000`), InputGain 25.
- **Audio DSP is on by default.** `DynamicProcessing.Enabled`, `BeamformingEnabled`, `EchoCancellingEnabled`, `NoiseCancellingEnabled` and `WindNoiseCancellingEnabled` are all `yes`.
  - Noise cancelling targets stationary noise, so it may attenuate train rumble.
  - Echo cancelling means playing train audio out of the intercom's *own* speaker will **not** reach the detector reliably. That is why the app has an `analyze.cgi?mode=inject` test hook.
  - Changing these settings is the operator's decision. The app does not touch them.
- `AllowUnsigned=false` by default on AXIS OS 12 (`/axis-cgi/applications/config.cgi?action=get&name=AllowUnsigned`). An unsigned `.eap` upload then fails with `Error:2`. It must be enabled by the operator before installing a dev build.
- SSH is disabled (`root.Network.SSH.Enabled=no`), so logs come via `/axis-cgi/admin/systemlog.cgi`.
- Bundled apps: Object Analytics, VMD and Barcode Reader, all stopped. The DLPU is unused, so no memory contention right now.

## Build host (Windows 11 26220, this machine)
- **Docker Desktop 4.76 crashes at startup.** Error: `initializing Inference manager: listening on unix://...\Docker\run\dockerInference: remove ...: The file cannot be accessed by the system`.
  - Stale AF_UNIX socket reparse points in `%LOCALAPPDATA%\Docker\run` cannot be deleted (`fsutil reparsepoint delete` fails with error 1920).
  - Renaming the folder did not help: the crash recurs immediately, even with `EnableInference=false`.
  - Earlier `run.stale-*` folders show this has happened before on this machine.
- **Workaround: use the native Docker engine inside the WSL `Ubuntu` distro** (Docker 29.1.3): `wsl -d Ubuntu -- bash -lc 'docker ...'`. Everything here builds that way.
- **Disk.** C: has only about 19–22 GB free, and the WSL vhdx also lives on C:. Keep datasets small. Resample everything to 16 kHz PCM16 straight away: 7,161 clips take about 2 GB.
- **ASan in containers.** It loops on `AddressSanitizer:DEADLYSIGNAL` under the WSL kernel's high mmap ASLR entropy. Fix: `setarch $(uname -m) -R`, which in turn needs `--security-opt seccomp=unconfined` on `docker run`.
- The ACAP SDK image has no native `gcc`, only the aarch64 cross toolchain. Host unit tests use a `debian:bookworm-slim` + gcc image (`tools/test_host.sh`).

## ACAP SDK 12.11 specifics
- Manifest schema up to **2.2.0**.
- `vendorId` is **required** and must match `^[A-Fa-f0-9]{10}$`. Use the value assigned by the ACAP Service Portal when signing; `0000000000` is a dev placeholder.
- `architecture` is required by the schema, but `acap-build` fills it in.
- Setting `compatibleOsVersions.min` below **12.11.72** (the SDK's floor) gives a warning. Binaries linked against the 12.11 SDK should not claim older OS support.
- `acap-build` fails with "Could not find a readable LICENSE file" unless `app/LICENSE` exists.
- Extra package files are added with `acap-build ./ -a model/traindet.tflite -a model/yamnet_class_map.csv`.
- Headers live in `/opt/axis/acapsdk/sysroots/aarch64/usr/include`. jansson, fcgi, liblarod and libpipewire-0.3 are all present.
- **AXEvent.** `ax_event_handler_declare2(handler, kvs, stateless=FALSE, "active", ...)` declares a stateful event with an explicit property key.
- **AXParameter.** Callbacks receive the fully qualified name `root.<App>.<Param>`. The group case may differ from `appName`, so strip the prefix case-insensitively. Do not call `ax_parameter_*` inside the callback.
- **CSP.** The device CSP (`script-src 'self'`) blocks inline `<script>` in settings pages, so JS goes in a separate `app.js`. Inline styles are allowed.
- **FastCGI.** All `httpConfig` `fastCgi` entries of an app share one `FCGI_SOCKET_NAME` socket, so dispatch on `SCRIPT_NAME`. Two worker threads keep `status.cgi` responsive while `analyze.cgi` runs.

## Data sourcing
- The HF mirror `Fhrozen/FSD50k` serves per-file WAVs without auth, but **rate-limits anonymous bulk downloads (HTTP 429)**. With 12 parallel workers, 860 of 5,161 requests failed. Four workers plus `Retry-After`/exponential backoff on 429 recovered all of them.

## On-device verification: 2026-10-15, v0.1.0 on the I8307-VE

Command: `python tools/device.py verify`. **All checks passed.**

- **Install.** `upload.cgi` returned OK, and `list.cgi` showed `Status="Running"` with `SignatureStatus="Unknown"` (unsigned).
- **Audio.** The PipeWire capture node is **`AudioDevice0Input0`** (`Audio/Source`). It negotiated **16 kHz mono F32**, so no in-app resampling is needed.
- **Larod.** Device `cpu-tflite`; the float32 TFLite model (16.9 MB) loaded first try.
  - Latency **135 ms average, 298 ms maximum** per 0.975 s window. That is about 28% of the 0.48 s hop budget.
  - 0 errors and 0 skipped windows.
- **Parity.** On-device vs host TFLite: max abs diff **5e-4** across 6 held-out clips. The small offset is CPU float/XNNPACK numerics.
- **Settings.** A `param.cgi` update reached the app live through the AXParameter callback; `status.cgi` showed the change in under 1.5 s.
  - Saving through `param.cgi` re-fires the callback for **every** app parameter, not just the changed one; see the syslog.
- **Event.** `GetEventInstances` lists `tnsaxis:CameraApplicationPlatform/TrainDetect/TrainDetected`. In v0.5.2 this was replaced by `TrainPresent`, `HornDetected` and `TrainPassed`; see "Action rules" below.
- **End-to-end.** Injected 28.7 s of held-out train audio (`analyze.cgi?mode=inject`). The run produced `TRAIN START score=0.84` at +4.8 s, an event, and `TRAIN END duration=26.0s peak=0.93`.
- **Ambient check (quiet room).** About −59 dBFS. The top classes were Silence, Noise and Hum, with a train score of about 0.04.
- **Not verified:**
  - a real train;
  - the settings page in a browser, which needs an interactive device login;
  - action rules firing on the event.

## v0.2.0 UI redesign (2026-10-15)
- **Fonts.** The device CSP (`default-src 'self'`) blocks Google Fonts, so the UI cannot load IBM Plex from a CDN.
  - The stack lists IBM Plex first, then falls back to system fonts.
  - Large numbers use the sans face with `font-variant-numeric: tabular-nums`. A missing mono font falls back to Courier, which looks wrong.
- **Cache-busting.** Link assets as `style.css?v=<version>` and `app.js?v=<version>`. Otherwise browsers keep the previous release's CSS/JS after an app upgrade; this was observed in the preview.
- **Upgrade start.** After uploading over a running app, `control.cgi?action=start` returns `Error: 6` because it is already running. That is harmless.
- **CSRF guard verified on device.**
  - `label.cgi` without `X-Requested-With` returns 403; a GET returns 405.
  - `analyze.cgi` with a non-`audio/wav` Content-Type returns 403.
- **Local UI iteration.** `tools/ui_preview.py` serves the HTML with a mock API, and `.claude/launch.json` holds the `ui-preview` config for the desktop app's browser pane.

## Security audit lessons (v0.4.0, 2026-10-15)
- **Never call `ax_parameter_set()` on the GLib main loop.**
  - It blocks on parhand, while parhand may be waiting for the main loop to accept a change callback. Under rapid `param.cgi` updates this deadlocks until the 25 s D-Bus timeout, and the `param.cgi` request hangs.
  - Use a worker thread instead.
- **One AXParameter handle per app.** A second `ax_parameter_new()` in the same process aborts with "Unable to register objects on DBus" (SIGABRT, then a respawn loop until systemd's start limit). Share the single handle.
- **Queued corrective writes must re-check the current value** before writing (compare-and-set), or they overwrite newer valid updates.
- **Out-of-range `int` params.** For `int:min;max` params, param.cgi clamps an out-of-range value, *stores* it and still answers `# Error`. `string` params are stored verbatim (`nan`, `1e308`, empty), so validate them in the app.
- **Apache routing.** AXIS OS 12.11 Apache returns 404 for any path after a FastCGI name (`label.cgi/analyze.cgi`). `SCRIPT_NAME` is always set. Route on the exact name anyway.
- **HTTPS auth.** Under the default policy, Digest over HTTPS gives 401; use Basic over HTTPS, as in `tools/device.py`.
- **ACAP crash loops.** systemd gives up after ~5 rapid failures ("Start request repeated too quickly") and the app shows Stopped. Redeploy or `control.cgi?action=start` to recover.
- **Key-sound timeline.** `timeline.cgi` keeps ~5 minutes of per-window group scores, which is the data source for tuning horn and speech fusion. The 1 Hz `status.cgi` polling misses sub-second horn blasts.

## AXEvent: stateful event verified (v0.5.0)
- **Declaration.** `ax_event_handler_declare2(..., stateless=FALSE, "active", ...)` with `active` marked as data produces `aev:isProperty="true"`, and `active` gets `isPropertyState="true"` in `GetEventInstances`. That is what the rules engine needs for "while active" conditions.
- **Observed on an ONVIF pull-point.**
  - On subscribe: `Initialized` carrying the current state.
  - On transitions: `Changed active=1` and `Changed active=0`.
  - Example: inject at +0 s, active at +4.5 s, inactive at +45.7 s for a 37.8 s clip with 8 s release.
- **Pull-point gotchas.**
  - `CreatePullPointSubscription` on `/vapix/services` returns the address `/onvif/services`, which requires a separate **ONVIF user** (401 for the root account).
  - `PullMessages` sent to `/vapix/services` with the `SubscriptionId` reference parameter (`xmlns="http://www.axis.com/2009/event"`, `wsa:IsReferenceParameter="true"`) works with the normal device login.

## Action rules with ACAP property events (v0.5.2)
- **Rule creation API.** `AddActionRule` (VAPIX action1 SOAP on `/vapix/services`) with the condition `TopicExpression` plus an ItemFilter, e.g. `boolean(//SimpleItem[@Name="active" and @Value="1"])`, and **no StartEvent** gives a "while" rule. The `com.axis.action.unlimited.*` templates act for as long as the condition holds.
- **Topic schemas are sticky.** **The event system keeps every schema a topic has ever been declared with, until reboot.** After the app changed `TrainDetected`'s data keys (v0.3 → v0.4 added `test`), `AddActionRule` failed with `InvalidConditionFilterFault: expression matched multiple property events`. Without a filter, it failed with "property state key's value not specified".
  - Fix: never change a declared topic's keys. Use a new topic name instead (now `TrainPresent`), or reboot.
  - Keep property events to the state key only, and move extra data into separate stateless events.
- **Overlay-text action.** `com.axis.action.unlimited.set_overlay` (params `text`, `channels`, `index`) writes into dynamic text slot `#D<index>`. A text overlay containing `#D1` must exist to see it; `/axis-cgi/dynamicoverlay/dynamicoverlay.cgi` `addText` / `remove`.
- **Verified with a temporary rule.** Overlay text appeared only while an injected train made TrainPresent active. The rule, action and overlay were removed afterwards (`tools/action_rules.py` lists rules).
