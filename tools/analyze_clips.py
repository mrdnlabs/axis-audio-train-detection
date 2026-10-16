"""Run WAV clips (16 kHz mono PCM16) through the on-device model (analyze.cgi) and
simulate the detector on the returned windows.

  python tools/analyze_clips.py <dir with .wav> <out.json>

Prints a per-clip summary: train-score stats, whether the detector (current device
settings) would fire, and the key-sound group peaks (rail, train horn, other horns,
road vehicles, speech, music).
"""
import glob
import io
import json
import os
import struct
import sys
import wave

sys.path.insert(0, os.path.dirname(__file__))
import device as d  # noqa: E402

CHUNK_S = 59
GROUPS = ["rail", "train_horn", "other_horn", "vehicle", "speech", "music"]


def chunks(path):
    with wave.open(path) as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1 and w.getsampwidth() == 2, path
        frames = w.readframes(w.getnframes())
    step = CHUNK_S * 16000 * 2
    for off in range(0, len(frames), step):
        part = frames[off:off + step]
        if len(part) < 3200:
            continue
        buf = io.BytesIO()
        with wave.open(buf, "wb") as o:
            o.setnchannels(1)
            o.setsampwidth(2)
            o.setframerate(16000)
            o.writeframes(part)
        yield off / 32000.0, buf.getvalue()


def thr_of(s):
    return 0.95 - 0.009 * (min(100, max(1, s)) - 1)


def horn_thr_of(s):
    return 0.6 - 0.0055 * (min(100, max(1, s)) - 1)


def fuse(wins, cfg):
    """Python mirror of fusion.c: sets w["fused"] and w["horn_eff"]."""
    rail_hist, levels = [], []
    for w in wins:
        g = w["groups"]
        rail_hist = (rail_hist + [max(g[0], g[1])])[-20:]
        recent, near = max(rail_hist[-6:]), max(rail_hist)
        bg = sorted(levels)[len(levels) // 5] if len(levels) >= 20 else w["level_db"]
        levels = (levels + [w["level_db"]])[-125:]
        h = w["prob"]
        if cfg.get("speech_suppression", 0) > 0:
            h *= 1 - cfg["speech_suppression"] * min(1, max(0, g[4] - recent))
        if cfg.get("require_rail"):
            h *= min(1, max(0, recent / 0.2))
        if cfg.get("min_rise_db", 0) > 0 and len(levels) > 21 and w["level_db"] < bg + cfg["min_rise_db"]:
            h = 0.0
        w["fused"] = min(1.0, max(0.0, h))
        w["horn_eff"] = g[2] if cfg.get("horn_includes_other_horns") and near >= 0.2 and g[2] > g[1] else g[1]


def simulate(wins, cfg):
    """Python mirror of fusion.c + detector.c (score source head)."""
    fuse(wins, cfg)
    thr, hthr = thr_of(cfg["sensitivity"]), horn_thr_of(cfg["horn_sensitivity"])
    n = max(1, min(64, round(cfg["smoothing_s"] / 0.48)))
    buf, events = [], []
    above = below = 0.0
    active, cool_until, horn_run = None, -1e9, 0
    for w in wins:
        gated = w["level_db"] < cfg["min_level_db"]
        buf = (buf + [0.0 if gated else w["fused"]])[-n:]
        s = sum(buf) / len(buf)
        horn_run = horn_run + 1 if (not gated and w["horn_eff"] >= hthr) else 0
        if active is None:
            above = above + 0.48 if s >= thr else 0.0
            if w["t"] >= cool_until and (above >= cfg["trigger_s"] or (cfg["horn_triggers"] and horn_run == 2)):
                active, below = w["t"], 0.0
        else:
            below = 0.0 if s >= thr else below + 0.48
            if below >= cfg["release_s"]:
                events.append((active, w["t"] - below))
                active, above, cool_until = None, 0.0, w["t"] + cfg["cooldown_s"]
    if active is not None:
        events.append((active, wins[-1]["t"]))
    return events


def main():
    src, out = sys.argv[1], sys.argv[2]
    cfg = d.status()["config"]
    results = {}
    for path in sorted(glob.glob(os.path.join(src, "*.wav"))):
        name = os.path.splitext(os.path.basename(path))[0]
        wins = []
        for t0, body in chunks(path):
            r = d.S.post(d.BASE + "/local/train_detect/analyze.cgi", data=body,
                         headers={"Content-Type": "audio/wav"}, timeout=180)
            r.raise_for_status()
            for w in r.json()["windows"]:
                w["t"] += t0
                wins.append(w)
        if not wins:
            continue
        ev = simulate(wins, cfg)
        g = {k: max(w["groups"][i] for w in wins) for i, k in enumerate(GROUPS)}
        thr = thr_of(cfg["sensitivity"])
        frac = sum(w["prob"] >= thr for w in wins) / len(wins)
        results[name] = {"windows": wins, "events": ev, "group_peaks": g}
        print(f"{name[:52]:52s} {wins[-1]['t'] + 0.975:5.0f}s  max {max(w['prob'] for w in wins):.2f} "
              f"above {frac:4.0%}  events {len(ev)}  rail {g['rail']:.2f} horn {g['train_horn']:.2f} "
              f"other {g['other_horn']:.2f} veh {g['vehicle']:.2f} speech {g['speech']:.2f} music {g['music']:.2f}", flush=True)
    json.dump({"config": cfg, "clips": results}, open(out, "w"))


if __name__ == "__main__":
    main()
