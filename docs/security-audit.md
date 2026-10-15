# Security audit and adversarial review: v0.4.0

_Date: 2026-10-15._

**Scope:** `acap/app` (C, manifest, web UI), `tools/`, `model/` and the git history. The audit is measured against [mrdnlabs/axis-acap-guidance](https://github.com/mrdnlabs/axis-acap-guidance): the security audit (audits 1–7), the development standards, and the manifest, auth-policy and signing guides.

**Method:**
1. Two independent reviews, one checklist-driven against the guidance and one adversarial.
2. Secret scanning: gitleaks v8.21.2 over the git history, plus pickaxe searches.
3. String scan of the built `.eap`.
4. **Live probes against the deployed app** on an AXIS I8307-VE (ARTPEC-8, AXIS OS 12.11).
5. Every fix was re-verified on the device.

## Verdict

**Credentials: none.** No credentials are hardcoded or committed:
- gitleaks over the full history found no leaks.
- `git log -S` for `password`, `secret`, `token`, `root:` and private IP ranges found nothing.
- The only credential-related strings in tracked files are the *variable names* `DEVICE_USER`/`DEVICE_PASS`. Their values come from the git-ignored `.env.devices`, which is used only by dev tools.
- The `.eap` contains no credential-like strings and no secret parameters.
- The app makes no VAPIX or outbound calls, so it needs no credentials at all.

**Remaining before any distribution:** the package is unsigned (placeholder `vendorId`), so installing needs the device-wide "Allow unsigned apps". Sign through the ACAP Signing Service (non-TIP route) and drop that step.

## Live probe results (final build)

| Probe | Result |
|---|---|
| Any endpoint or page without credentials | 401 |
| Extra path segments, e.g. `label.cgi/analyze.cgi` or `status.cgi/x` | 404 from Apache, and the app also rejects any `PATH_INFO` |
| `analyze.cgi` with a simple CSRF content type, `Origin: https://evil.example` or `Origin: null` | 403 |
| GET on state-changing endpoints | 405 |
| `label.cgi` without `X-Requested-With` | 403 |
| WAV: wrong rate or channels, garbage, header only, lying `data` size (2 GB), more than 60 s, lying `Content-Length` | 415 or 413, or clamped safely. No crash. |
| `label.cgi` with `start=nan`, `inf`, `1e400`, negative, 5,000 digits, or an unknown label | 400 or 404 |
| Parameter fuzzing: `nan`, `-inf`, `1e308`, empty, `abc`, `3s`, out-of-range ints, a bad enum | Ignored or clamped. The store is corrected to the effective value, and the app stays healthy. |
| 55 rapid back-to-back `param.cgi` updates | Slowest 0.09 s. No stall. Valid values sent afterwards are kept. |
| `X-Frame-Options` / `nosniff` / `Cache-Control` | `SAMEORIGIN`, `nosniff`, `no-store` |

## Findings and status

| # | Severity | Finding | Status |
|---|---|---|---|
| 1 | Medium | **Inference failure (larod restart) never recovered, and an active event could stay latched forever.** | **Fixed.** After ~10 s of consecutive failures, or 60 s without audio that had been streaming, the app ends the active event (`active=false` is sent) and exits through the main loop for respawn. |
| 2 | Medium | **Injected test audio produced events indistinguishable from real ones.** | **Fixed.** A `test` data key is added to the AXEvent and to history and the CSV. The UI asks for confirmation before injecting. |
| 3 | Medium | **No maximum event duration.** Persistent noise above the threshold latched "while active" rules. | **Fixed.** `MaxEventSeconds` (default 600) force-ends the event, and it is flagged `capped`. |
| 4 | Medium | **Crash or restart loops.** The app restarted every 60 s on devices with no microphone audio, and had a tight loop on startup failures. | **Fixed.** "No audio" is reported in status instead of exiting. Startup failures back off for 30 s. |
| 5 | Medium | **Routing used substring matching on `SCRIPT_NAME`, with a `REQUEST_URI` fallback.** In theory an operator-to-admin path existed. | **Not exploitable on AXIS OS 12.11** (Apache returns 404 for extra path, as verified). **Hardened anyway:** exact CGI-name routing, `PATH_INFO` rejected, no fallback. |
| 6 | Low–Medium | **String-typed seconds params.** `""` or `abc` became the most sensitive setting, and `nan` stayed in the store while the app used another value. | **Fixed.** Strict parsing keeps the previous value on bad input, and a worker thread writes the effective value back to the store. |
| 6a | High *(regression found during this audit)* | **The first write-back implementation deadlocked against the parameter daemon.** It called `ax_parameter_set` on the main loop while `parhand` waited for that loop: 25 s D-Bus timeouts under rapid updates. A second AXParameter handle aborted the app instead ("Unable to register objects on DBus"). A later version also let stale corrections overwrite newer valid values. | **Fixed and re-verified.** Writes now happen on a worker thread that shares the app's single handle, after a 300 ms settle, and only if the store still holds the bad value. |
| 7 | Low (privacy) | **Viewers could see live sound classifications** (for example, Speech). | **Fixed.** `status.cgi` and `timeline.cgi` now require operator access. |
| 8 | Low | **The Test-tab verdict could disagree with the device.** It ignored horn-triggered starts and used time-based smoothing. | **Fixed.** The JS simulator mirrors `detector.c` (count-based smoothing, horn starts, cooldown, cap). The Tuning preview uses the exact 0.48 s timeline. |
| 9 | Low | **A mid-event horn re-sent `active=true`,** re-firing "on event" rules. | **Fixed.** No mid-event re-send. The horn flag is reported on START and END. |
| 10 | Low | **After a stall, one window could count as up to 5 s of trigger evidence.** | **Fixed.** `dt` is capped at 1 s. |
| 11 | Low | **Event durations used the wall clock,** so an NTP step could corrupt them. | **Fixed.** Durations are monotonic. Host test included. |
| 12 | Low | **The realtime audio callback contended with frees of 3.8 MB inject buffers** under the same lock. | **Fixed.** Pointers are swapped under the lock and freed after unlocking. |
| 13 | Low | **The UI could freeze** (fetch had no timeout), saves could arrive out of order, and settings-load error detection was too broad. | **Fixed.** 5 s fetch timeouts, serialized saves, and an exact `# Error` match. |
| 14 | Low | **`exit()` was called from a worker thread.** | **Fixed.** Exit happens through the main loop (`failsafe_exit`). |
| 15 | Low | **Dev tool used cleartext HTTP.** | **Fixed.** `tools/device.py` defaults to HTTPS with Basic auth, per the auth-policy guide; `DEVICE_CA` enables certificate verification. The device itself still accepts HTTP. Enabling HTTPS-only is a device setting, recommended for deployment. |
| 16 | Low | **Documentation drift** (endpoints, `.eap` name, signing note). | **Fixed.** |
| 17 | Info | **FastCGI socket mode 0777** (SDK example pattern). Any local process could bypass Apache auth. | **Accepted and documented.** The socket lives in the app's runtime directory, and local processes are inside the device trust boundary. |
| 18 | Info | **The Origin check rejects a `Host` that carries an explicit default port.** | **Accepted.** It fails closed. |
| 19 | Publish blocker | **Unsigned package and placeholder `vendorId`.** | **Open.** Sign before distribution. |

## Checklist summary (guidance audits 1–7)

1. **Network exposure: PASS.** FastCGI only, with the least access level per endpoint. No ports or anonymous access. UI paths are app-relative.
2. **Credentials: PASS.** None in code, history or the package. No VAPIX use. `.env.devices` is git-ignored.
3. **Input validation: PASS** after fixes #5 and #6.
   - No `system`, `popen`, `sprintf` or `strcpy`.
   - GET has no side effects.
   - CSRF guards are in place.
   - Every dynamic UI value is written with `textContent`.
4. **Build and packaging: PASS**, except signing (#19).
   - WSL chmod fix, `.dockerignore`, `--no-cache`, schema 2.2.0, `respawn`.
   - LF line endings in the index.
   - The `.eap` includes the Apache-2.0 text.
5. **Publish readiness: PASS**, except signing.
   - No TODO markers or dead code.
   - README and requirements are accurate.
   - Tested on the device.
6. **Memory and resources: PASS.** Bounded rings (audio 8 s, timeline 625, history 50). The UI series is capped. Shutdown and fail-safe paths are documented.
7. **Concurrency: PASS** after fix #6a.
   - Contexts are documented: main loop, inference, 2 FastCGI workers, param write-back worker, and the PipeWire loop plus its realtime callback.
   - Each shared variable has one owning lock.
   - No lock nesting, and no app lock is held across I/O.

## Device-side recommendations (operator decisions, not app changes)
- **HTTPS only.** Enable HTTPS-only on the device, so the UI and API are not reachable over cleartext HTTP.
- **Unsigned apps.** Turn "Allow unsigned apps" off again once a signed build is installed.
- **Password.** Rotate the lab device's root password, which was shared during development.
