"""Local preview of the settings UI with a mock device API (no device needed).

  python tools/ui_preview.py [port]      -> http://localhost:8765/#live

Serves acap/app/html/ and fakes status.cgi, param.cgi, analyze.cgi, label.cgi and
basicdeviceinfo.cgi with synthetic data: a train passes every 90 s.
"""
import json
import math
import os
import sys
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "acap", "app", "html"))
T0 = time.time()
PARAMS = {"Enabled": "yes", "Sensitivity": "50", "TriggerSeconds": "3", "ReleaseSeconds": "8",
          "SmoothingSeconds": "2", "MinLevelDbfs": "-65", "CooldownSeconds": "10", "HornSensitivity": "50",
          "HornTriggers": "yes", "ScoreSource": "head", "MaxEventSeconds": "600"}
HISTORY = [
    {"start": T0 - 3600 * h - 600, "end": T0 - 3600 * h - 600 + d, "duration_s": d, "peak_score": p,
     "mean_score": round(p - 0.3, 3), "peak_level_db": lv, "horn": hn, "label": None}
    for h, d, p, lv, hn in [(1, 26.0, 0.93, -17.3, False), (2.3, 41.5, 0.88, -15.8, True), (3.8, 18.2, 0.71, -24.0, False),
                            (5.1, 63.0, 0.95, -12.6, True), (7.4, 35.4, 0.9, -16.1, True)]
]


def thr(s):
    return 0.95 - 0.009 * (int(s) - 1)


def groups_at(t):
    ph = (t - T0) % 90
    act = 20 <= ph < 50
    return [0.7 + 0.2 * math.sin(t) if act else 0.02,                     # rail
            0.55 if 30 <= ph < 31.5 else 0.01,                            # train horn
            0.7 if 34 <= ph < 35 else 0.02,                               # other horns
            0.6 if 5 <= ph < 15 else (0.3 if act else 0.03),             # road vehicles
            0.85 if 60 <= ph < 75 else 0.02,                              # speech
            0.4 if 76 <= ph < 85 else 0.01]                               # music


def timeline(since):
    now = time.time()
    newest = int((now - T0) / 0.48) + 1
    rows = []
    for seq in range(max(since + 1, newest - 624, 1), newest + 1):
        t = T0 + (seq - 1) * 0.48
        ph = (t - T0) % 90
        act = 20 <= ph < 50
        sc = 0.6 + 0.3 * math.sin(math.pi * (ph - 20) / 30) if act else 0.05
        rows.append([seq, t, sc, sc, -20.0 if act else -60.0, 2 if act else 0] + groups_at(t))
    return {"seq": newest, "hop_s": 0.48, "groups": ["rail", "train_horn", "other_horn", "vehicle", "speech", "music"], "rows": rows}


def status():
    t = time.time()
    phase = (t - T0) % 90
    active = 20 <= phase < 50
    score = 0.6 + 0.3 * math.sin(math.pi * (phase - 20) / 30) if active else 0.05 + 0.04 * abs(math.sin(t))
    level = -20 - 4 * math.sin(t) if active else -60 + 3 * math.sin(t / 3)
    top = ([("Train", 0.81), ("Rail transport", 0.74), ("Railroad car, train wagon", 0.52), ("Vehicle", 0.31), ("Rumble", 0.18)]
           if active else [("Silence", 0.42), ("Inside, small room", 0.12), ("Hum", 0.06), ("Mechanisms", 0.04), ("Printer", 0.03)])
    cur = None
    if active:
        start = t - (phase - 20)
        cur = {"start": start, "end": 0, "duration_s": 0, "peak_score": 0.9, "mean_score": 0.7, "peak_level_db": -17,
               "horn": phase > 30, "label": None, "elapsed_s": phase - 20}
    return {
        "app": "train_detect", "state": "active" if active else "idle", "uptime_s": t - T0 + 11000, "current": cur,
        "scores": {"head": score, "yamnet_rail": score * 0.7, "horn": 0.5 if 30 < phase < 32 else 0.01, "score": score,
                   "threshold": thr(PARAMS["Sensitivity"]), "horn_threshold": 0.6 - 0.0055 * (int(PARAMS["HornSensitivity"]) - 1),
                   "level_db": level, "gated": level < int(PARAMS["MinLevelDbfs"]), "horn_now": False,
                   "top": [{"index": i, "name": n, "score": s} for i, (n, s) in enumerate(top)]},
        "history": sorted(HISTORY, key=lambda e: -e["start"]),
        "last_horn": t - (phase - 30) if active and phase > 30 else None,
        "inference": {"count": int((t - T0) * 2), "errors": 0, "skipped_windows": 0, "latency_ms_avg": 107.3,
                      "latency_ms_max": 298.2, "last_wall": t, "device": "cpu-tflite"},
        "audio": {"node": "AudioDevice0Input0", "streaming": True, "native_rate": 16000, "native_channels": 1,
                  "samples": 0, "last_data_age_s": 0.02, "inject_remaining_s": 0.0},
        "config": {"enabled": PARAMS["Enabled"] == "yes", "sensitivity": int(PARAMS["Sensitivity"]),
                   "threshold": thr(PARAMS["Sensitivity"]), "trigger_s": float(PARAMS["TriggerSeconds"]),
                   "release_s": float(PARAMS["ReleaseSeconds"]), "smoothing_s": float(PARAMS["SmoothingSeconds"]),
                   "horn_sensitivity": int(PARAMS["HornSensitivity"]), "horn_triggers": PARAMS["HornTriggers"] == "yes",
                   "min_level_db": float(PARAMS["MinLevelDbfs"]), "score_source": PARAMS["ScoreSource"],
                   "cooldown_s": float(PARAMS["CooldownSeconds"]), "horn_threshold": 0.33},
        "events_total": len(HISTORY), "horns_total": 3,
    }


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=ROOT, **kw)

    def log_message(self, *a):
        pass

    def send_json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_text(self, s):
        body = s.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def handle_api(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        if u.path.endswith("/timeline.cgi"):
            return self.send_json(timeline(int(q.get("since", 0))))
        if u.path.endswith("/status.cgi"):
            return self.send_json(status())
        if u.path == "/axis-cgi/param.cgi":
            if q.get("action") == "list":
                return self.send_text("".join(f"root.Train_detect.{k}={v}\n" for k, v in PARAMS.items()))
            for k, v in q.items():
                if k.startswith("root.Train_detect."):
                    PARAMS[k.split(".")[-1]] = v
            return self.send_text("OK")
        if u.path == "/axis-cgi/basicdeviceinfo.cgi":
            return self.send_json({"data": {"propertyList": {"ProdShortName": "AXIS I8307-VE"}}})
        if u.path.endswith("/label.cgi"):
            for e in HISTORY:
                if abs(e["start"] - float(q.get("start", 0))) < 0.01:
                    e["label"] = None if q.get("label") == "none" else q.get("label")
                    return self.send_json({"start": e["start"]})
            return self.send_json({"error": "no such event"}, 404)
        if u.path.endswith("/analyze.cgi"):
            n = int(self.headers.get("Content-Length", 0))
            self.rfile.read(n)
            dur = max(1.0, (n - 44) / 32000)
            if q.get("mode") == "inject":
                return self.send_json({"mode": "inject", "duration_s": dur})
            wins = []
            k = max(1, int((dur - 0.975) / 0.48) + 1)
            for i in range(k):
                f = i / max(1, k - 1)
                p = 0.6 + 0.33 * math.sin(math.pi * f) if 0.1 < f < 0.95 else 0.1
                wins.append({"t": i * 0.48, "prob": p, "rail": p * 0.7, "horn": 0.02, "level_db": -20.0, "latency_ms": 131,
                             "top": [{"name": "Train", "score": p}, {"name": "Rail transport", "score": p * 0.8},
                                     {"name": "Rumble", "score": 0.2}],
                             "groups": [p * 0.8, 0.5 if i in (8, 9) else 0.01, 0.6 if i == 14 else 0.02, 0.2, 0.05, 0.01]})
            return self.send_json({"duration_s": dur, "windows_n": k, "max_prob": max(w["prob"] for w in wins),
                                   "mean_prob": sum(w["prob"] for w in wins) / k, "windows": wins})
        return False

    def do_GET(self):
        if self.handle_api() is False:
            super().do_GET()

    def do_POST(self):
        if self.handle_api() is False:
            self.send_json({"error": "not found"}, 404)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    print(f"preview on http://localhost:{port}/#live")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
