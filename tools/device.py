"""Deploy and verify train_detect on an Axis device.

Credentials come from .env.devices (git-ignored): DEVICE_IP, DEVICE_USER, DEVICE_PASS.

  python tools/device.py deploy build/train_detect_0_1_0_aarch64.eap
  python tools/device.py verify            # health, model parity, settings, live event
  python tools/device.py status            # one status snapshot
  python tools/device.py logs [N]          # last N [traindet] syslog lines
"""
import glob
import json
import os
import re
import sys
import time

import requests
from requests.auth import HTTPDigestAuth

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
APP = "train_detect"


def env():
    vals = {}
    with open(os.path.join(ROOT, ".env.devices"), encoding="utf-8") as f:
        for line in f:
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.strip().split("=", 1)
                vals[k] = v
    return vals


E = env()
BASE = f"http://{E['DEVICE_IP']}"
AUTH = HTTPDigestAuth(E["DEVICE_USER"], E["DEVICE_PASS"])
S = requests.Session()
S.auth = AUTH


def get(path, **kw):
    return S.get(BASE + path, timeout=kw.pop("timeout", 20), **kw)


def ok(msg):
    print(f"  PASS  {msg}")


def bad(msg):
    print(f"  FAIL  {msg}")
    return 1


def deploy(eap):
    r = get("/axis-cgi/applications/config.cgi?action=get&name=AllowUnsigned")
    if 'value="true"' not in r.text:
        print("Device does not allow unsigned apps (AllowUnsigned=false).\n"
              "Enable it in the device web UI (Apps > Allow unsigned apps) and retry.")
        return 2
    print(f"uploading {eap} ...")
    with open(eap, "rb") as f:
        r = S.post(BASE + "/axis-cgi/applications/upload.cgi", files={"packfil": (os.path.basename(eap), f)},
                   timeout=300)
    print("upload:", r.status_code, r.text.strip()[:200])
    if r.status_code != 200 or "OK" not in r.text:
        return 1
    r = get(f"/axis-cgi/applications/control.cgi?action=start&package={APP}")
    print("start:", r.status_code, r.text.strip()[:200])
    time.sleep(3)
    r = get("/axis-cgi/applications/list.cgi")
    m = re.search(rf'<application Name="{APP}"[^>]*', r.text)
    print(m.group(0) if m else "app not listed!")
    return 0


def status():
    r = get(f"/local/{APP}/status.cgi")
    r.raise_for_status()
    return r.json()


def logs(n=40):
    r = get("/axis-cgi/admin/systemlog.cgi")
    if r.status_code != 200:
        r = get("/axis-cgi/systemlog.cgi")
    lines = [l for l in r.text.splitlines() if "traindet" in l or APP in l]
    return lines[-n:]


def wav_post(path, mode=None):
    with open(path, "rb") as f:
        body = f.read()
    url = f"{BASE}/local/{APP}/analyze.cgi" + (f"?mode={mode}" if mode else "")
    return S.post(url, data=body, headers={"Content-Type": "audio/wav"}, timeout=180)


def param_group():
    for g in ("root.Train_detect", "root.train_detect"):
        t = get(f"/axis-cgi/param.cgi?action=list&group={g}").text
        if "=" in t and "rror" not in t:
            return t.splitlines()[0].split(".")[0] + "." + t.splitlines()[0].split(".")[1]
    raise RuntimeError("param group not found")


def set_params(**kv):
    g = param_group()
    q = {"action": "update"}
    q.update({f"{g}.{k}": str(v) for k, v in kv.items()})
    r = get("/axis-cgi/param.cgi", params=q)
    return r.text.strip()


def event_declared():
    soap = ('<?xml version="1.0" encoding="UTF-8"?><SOAP-ENV:Envelope '
            'xmlns:SOAP-ENV="http://www.w3.org/2003/05/soap-envelope" '
            'xmlns:aev="http://www.axis.com/vapix/ws/event1"><SOAP-ENV:Body>'
            '<aev:GetEventInstances/></SOAP-ENV:Body></SOAP-ENV:Envelope>')
    r = S.post(BASE + "/vapix/services", data=soap, headers={"Content-Type": "application/soap+xml"}, timeout=30)
    return "TrainDetect" in r.text, r.status_code


def verify():
    fails = 0
    print("== health")
    s1 = status()
    time.sleep(3)
    s2 = status()
    a, inf = s2["audio"], s2["inference"]
    fails += 0 if a["streaming"] else bad(f"audio not streaming: {a}")
    if a["streaming"]:
        ok(f"audio streaming from {a['node']} @ {a['native_rate']} Hz x{a['native_channels']}")
    n = inf["count"] - s1["inference"]["count"]
    fails += 0 if n >= 4 else bad(f"only {n} inferences in 3 s")
    if n >= 4:
        ok(f"{n} inferences in 3 s on {inf['device']}, latency avg {inf['latency_ms_avg']:.0f} ms "
           f"max {inf['latency_ms_max']:.0f} ms, errors {inf['errors']}, skipped {inf['skipped_windows']}")
    fails += 0 if inf["latency_ms_avg"] < 300 else bad("latency over budget")
    print(f"  info  level {s2['scores']['level_db']:.1f} dBFS, score {s2['scores']['score']:.3f}, "
          f"top: {[t['name'] for t in s2['scores']['top']]}")

    print("== model parity (on-device vs host TFLite)")
    tv = os.path.join(ROOT, "model", "out", "testvec")
    expected = json.load(open(os.path.join(tv, "expected.json")))
    worst = 0.0
    for name, exp in expected.items():
        r = wav_post(os.path.join(tv, name + ".wav"))
        if r.status_code != 200:
            fails += bad(f"{name}: HTTP {r.status_code} {r.text[:200]}")
            continue
        got = r.json()["windows"]
        if len(got) != len(exp["windows"]):
            fails += bad(f"{name}: {len(got)} windows vs {len(exp['windows'])} expected")
            continue
        d = max(abs(g["prob"] - e["prob"]) for g, e in zip(got, exp["windows"]))
        dr = max(abs(g["rail"] - e["rail"]) for g, e in zip(got, exp["windows"]))
        worst = max(worst, d, dr)
        mp = max(g["prob"] for g in got)
        print(f"  info  {name:28s} label={exp['label']} max_prob={mp:.3f} max|diff| prob={d:.2e} rail={dr:.2e}")
    fails += 0 if worst < 0.02 else bad(f"parity diff {worst:.3f} >= 0.02")
    if worst < 0.02:
        ok(f"on-device scores match host TFLite (max abs diff {worst:.2e})")

    print("== settings take effect live")
    orig = status()["config"]["sensitivity"]
    print("  info  param.cgi update:", set_params(Sensitivity=77))
    time.sleep(1.5)
    c = status()["config"]
    fails += 0 if c["sensitivity"] == 77 else bad(f"sensitivity not applied: {c['sensitivity']}")
    if c["sensitivity"] == 77:
        ok(f"Sensitivity=77 applied live (threshold {c['threshold']})")
    set_params(Sensitivity=orig)

    print("== event declaration")
    found, code = event_declared()
    fails += 0 if found else bad(f"TrainDetect event not in GetEventInstances (HTTP {code})")
    if found:
        ok("tnsaxis:CameraApplicationPlatform/TrainDetect/TrainDetected declared")

    print("== end-to-end: inject a train recording into the live pipeline")
    pos = [os.path.join(tv, "e2e_train.wav")] if os.path.exists(os.path.join(tv, "e2e_train.wav")) else []
    if pos:
        before = status()["events_total"]
        r = wav_post(pos[0], mode="inject")
        print("  info  inject:", r.status_code, r.text[:120])
        seen_active = False
        for _ in range(40):
            time.sleep(1)
            s = status()
            seen_active |= s["state"] == "active"
            if s["events_total"] > before and s["state"] != "active" and s["audio"]["inject_remaining_s"] == 0:
                break
        s = status()
        if s["events_total"] > before and seen_active:
            ok(f"event fired and ended; newest history entry: {s['history'][0] if s['history'] else None}")
        else:
            fails += bad(f"no event from injected train audio (state {s['state']}, "
                         f"score {s['scores']['score']}, events {s['events_total']})")

    print("== syslog")
    for line in logs(15):
        print("   ", line[-160:])
    print(f"\n{'ALL CHECKS PASSED' if not fails else f'{fails} CHECK(S) FAILED'}")
    return 1 if fails else 0


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    cmd = argv[1]
    if cmd == "deploy":
        return deploy(argv[2] if len(argv) > 2 else sorted(glob.glob(os.path.join(ROOT, "build", "*aarch64.eap")))[-1])
    if cmd == "verify":
        return verify()
    if cmd == "status":
        print(json.dumps(status(), indent=1))
        return 0
    if cmd == "logs":
        print("\n".join(logs(int(argv[2]) if len(argv) > 2 else 40)))
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
