"""Evaluate interpretable fusion rules on top of the trained head.

Data (all per 0.48 s window):
  * held-out val/test splits (out/embeddings.npz, original audio only): head prob from
    out/head.h5, key-sound groups from YAMNet's classifier, level from rms_db
  * on-device results for open web train clips (out/webclips_results.json)
  * on-device live timeline snapshots containing speech (out/live_timeline_*.json)

Rules (each window, then the device's smoothing/trigger/release logic):
  rail_gate:   score = head * min(1, rail_recent / R0), rail_recent = max of
               max(rail, train_horn) over the last K windows
  speech_sup:  score *= 1 - A * max(0, speech - rail_recent)
  emergence:   window counts only if level >= background + E dB (background = 20th
               percentile of the previous 60 s; for short clips: of the clip)
Writes out/fusion_eval.json and prints a comparison + per-category effect table.
"""
import csv
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "yamnet"))
from common import DATA, MANIFEST, OUT  # noqa: E402

HOP = 0.48
G = ["rail", "train_horn", "other_horn", "vehicle", "speech", "music"]
IDX = {"rail": [322, 323, 326, 327, 328], "train_horn": [324, 325], "other_horn": [302, 312, 395, 397],
       "vehicle": [294, 300, 301, 308, 310, 315, 321, 337, 343, 346], "speech": [0, 1, 2, 3, 4, 5], "music": [132]}


def thr_of(s):
    return 0.95 - 0.009 * (min(100, max(1, s)) - 1)


def fuse(head, groups, level, rule):
    """Per-window fused score for one clip/stream. groups: (n, 6)."""
    n = len(head)
    score = head.astype(np.float64).copy()
    rail_any = np.maximum(groups[:, 0], groups[:, 1])
    if rule.get("rail_gate") or rule.get("speech_sup"):
        k = rule.get("K", 6)
        recent = np.array([rail_any[max(0, i - k + 1):i + 1].max() for i in range(n)])
    if rule.get("rail_gate"):
        score *= np.minimum(1.0, recent / rule["R0"])
    if rule.get("speech_sup"):
        score *= 1.0 - rule["A"] * np.clip(groups[:, 4] - recent, 0, 1)
    if rule.get("E") is not None:
        bg = np.array([np.percentile(level[max(0, i - 125):i + 1] if i >= 20 else level, 20) for i in range(n)])
        score = np.where(level >= bg + rule["E"], score, 0.0)
    return score


def detect(score, level, horn_ok, cfg):
    """Mirror of detector.c on a per-window score stream. Returns list of (start_i, end_i)."""
    thr = thr_of(cfg["sens"])
    n_s = max(1, min(64, round(cfg["smooth"] / HOP)))
    buf, ev = [], []
    above = below = 0.0
    active, cool, run = None, -1e9, 0
    for i, v in enumerate(score):
        gated = level[i] < cfg["min_level"]
        buf = (buf + [0.0 if gated else v])[-n_s:]
        s = sum(buf) / len(buf)
        run = run + 1 if (horn_ok[i] and not gated) else 0
        t = i * HOP
        if active is None:
            above = above + HOP if s >= thr else 0.0
            if t >= cool and (above >= cfg["trigger"] or (cfg["horn"] and run == 2)):
                active, below = i, 0.0
        else:
            below = 0.0 if s >= thr else below + HOP
            if below >= cfg["release"]:
                ev.append((active, i))
                active, above, cool = None, 0.0, t + cfg["cool"]
    if active is not None:
        ev.append((active, len(score) - 1))
    return ev


def horn_flags(groups, rule, hthr=0.33):
    th = groups[:, 1] >= hthr
    if rule.get("horn_conj"):
        k = 20  # other horns count when rail evidence is within ~10 s
        rail_any = np.maximum(groups[:, 0], groups[:, 1])
        near = np.array([rail_any[max(0, i - k):i + k + 1].max() for i in range(len(groups))])
        th = th | ((groups[:, 2] >= 0.5) & (near >= 0.2))
    return th


def load_testset():
    import tensorflow as tf  # noqa: F401
    from tf_keras import models
    sys.path.insert(0, os.path.dirname(__file__))
    from importlib import import_module
    th = import_module("03_train_head")
    w, b = th.yamnet_classifier()
    d = dict(np.load(os.path.join(OUT, "embeddings.npz")))
    m = d["aug"] == 0
    d = {k: v[m] for k, v in d.items()}
    emb = d["emb"].astype(np.float32)
    scores = 1.0 / (1.0 + np.exp(-np.clip(emb @ w + b, -60, 60)))
    head = models.load_model(os.path.join(OUT, "head.h5"), compile=False).predict(
        np.concatenate([emb, scores.astype(np.float32)], 1), batch_size=4096, verbose=0).ravel()
    groups = np.stack([scores[:, IDX[g]].max(1) for g in G], 1)
    rows = list(csv.DictReader(open(MANIFEST, encoding="utf-8")))
    clips = {}
    for c in np.unique(d["clip"]):
        mm = d["clip"] == c
        clips[int(c)] = dict(split=int(d["split"][mm][0]), label=int(rows[c]["label"]), labels=rows[c]["labels"],
                             head=head[mm], groups=groups[mm], level=d["rms_db"][mm].astype(np.float32))
    return clips


def load_web():
    r = json.load(open(os.path.join(OUT, "webclips_results.json")))
    out = {}
    for name, c in r["clips"].items():
        if "Whistle" in name:
            continue  # whistle-only recordings, not pass-bys
        ws = c["windows"]
        out[name] = dict(head=np.array([w["prob"] for w in ws]), groups=np.array([w["groups"] for w in ws]),
                         level=np.array([w["level_db"] for w in ws]))
    return out


def load_live():
    streams = []
    for p in sorted(glob.glob(os.path.join(OUT, "live_timeline_*.json"))):
        rows = json.load(open(p))["rows"]
        streams.append(dict(head=np.array([r[2] for r in rows]), groups=np.array([r[6:12] for r in rows]),
                            level=np.array([r[4] for r in rows])))
    return streams


def evaluate(rule, cfg, test, web, live, split):
    tp = fp = fn = 0
    conf = {}
    for c in test.values():
        if c["split"] != split:
            continue
        s = fuse(c["head"], c["groups"], c["level"], rule)
        fired = bool(detect(s, c["level"], horn_flags(c["groups"], rule), cfg))
        if c["label"]:
            tp += fired
            fn += not fired
        else:
            fp += fired
            for lab in ("Speech", "Male_speech_and_man_speaking", "Female_speech_and_woman_speaking", "Conversation",
                        "Music", "Truck", "Bus", "Aircraft", "Engine", "Thunder", "Wind"):
                if lab in c["labels"].split(","):
                    conf.setdefault(lab, []).append(fired)
    web_hits = sum(bool(detect(fuse(c["head"], c["groups"], c["level"], rule), c["level"],
                               horn_flags(c["groups"], rule), cfg)) for c in web.values())
    live_events = sum(len(detect(fuse(s["head"], s["groups"], s["level"], rule), s["level"],
                                 horn_flags(s["groups"], rule), cfg)) for s in live)
    prec = tp / (tp + fp) if tp + fp else 0
    rec = tp / (tp + fn) if tp + fn else 0
    return dict(precision=round(prec, 3), recall=round(rec, 3), f1=round(2 * prec * rec / (prec + rec), 3) if prec + rec else 0,
                tp=tp, fp=fp, fn=fn, web_recall=f"{web_hits}/{len(web)}", live_speech_events=live_events,
                confuser_fp={k: f"{sum(v)}/{len(v)}" for k, v in conf.items()})


def category_table(test, web, live):
    """Mean head score in windows where each group is dominant-ish (>= 0.5)."""
    def agg(streams, lab=None):
        res = {}
        for gi, g in enumerate(G):
            hs = np.concatenate([s["head"][s["groups"][:, gi] >= 0.5] for s in streams if (lab is None or s.get("label") == lab)] or [np.array([])])
            res[g] = (len(hs), float(hs.mean()) if len(hs) else float("nan"))
        return res
    tst = [c for c in test.values() if c["split"] == 2]
    return {"test_trains": agg(tst, 1), "test_non_trains": agg(tst, 0), "web_trains": agg(list(web.values())),
            "live_device": agg(live)}


def main():
    test, web, live = load_testset(), load_web(), load_live()
    base_cfg = dict(sens=50, smooth=2.0, trigger=3.0, release=8.0, cool=10.0, min_level=-65.0, horn=True)
    short_cfg = dict(base_cfg, trigger=1.5)  # FSD clips are often only a few seconds long
    rules = {
        "baseline (head only)": {},
        "rail_gate R0=0.2": {"rail_gate": True, "R0": 0.2, "K": 6},
        "rail_gate R0=0.3 K=10": {"rail_gate": True, "R0": 0.3, "K": 10},
        "speech_sup A=1": {"speech_sup": True, "A": 1.0, "K": 6},
        "rail_gate+speech_sup": {"rail_gate": True, "R0": 0.2, "K": 6, "speech_sup": True, "A": 1.0},
        "rail_gate+speech_sup+horn_conj": {"rail_gate": True, "R0": 0.2, "K": 6, "speech_sup": True, "A": 1.0,
                                           "horn_conj": True},
        "rail_gate+speech_sup+horn_conj+E6": {"rail_gate": True, "R0": 0.2, "K": 6, "speech_sup": True, "A": 1.0,
                                              "horn_conj": True, "E": 6.0},
    }
    out = {"categories": category_table(test, web, live), "rules": {}}
    print("\nMean train score (head) in windows where a group scores >= 0.5  [n windows, mean head]")
    for k, v in out["categories"].items():
        print(f"  {k:16s} " + "  ".join(f"{g}: {n:5d} {m:.2f}" for g, (n, m) in v.items()))
    print("\nRule comparison (val split tunes, test split reports; trigger 1.5 s for short FSD clips, "
          "web/live use the device default 3 s)")
    for name, rule in rules.items():
        r_val = evaluate(rule, short_cfg, test, web, live, split=1)
        r = evaluate(rule, short_cfg, test, web, live, split=2)
        r["web_recall_3s"] = evaluate(rule, base_cfg, test, web, live, split=2)["web_recall"]
        r["live_speech_events_3s"] = evaluate(rule, base_cfg, test, web, live, split=2)["live_speech_events"]
        out["rules"][name] = {"val": r_val, "test": r}
        print(f"  {name:38s} test P {r['precision']:.2f} R {r['recall']:.2f} F1 {r['f1']:.2f} (fp {r['fp']}) | "
              f"val F1 {r_val['f1']:.2f} | web {r['web_recall_3s']} | live speech events {r['live_speech_events_3s']} | "
              f"speech FP {r['confuser_fp'].get('Speech', '-')}  music FP {r['confuser_fp'].get('Music', '-')}  "
              f"truck {r['confuser_fp'].get('Truck', '-')}  aircraft {r['confuser_fp'].get('Aircraft', '-')}")
    json.dump(out, open(os.path.join(OUT, "fusion_eval.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
