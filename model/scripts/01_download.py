"""Download the first-swing training data and store it as 16 kHz mono PCM16 WAV.

Sources
  * ESC-50 (GitHub zip): 40 `train` clips + 1960 diverse negatives.
  * FSD50K (per-file via the Hugging Face mirror Fhrozen/FSD50k): every clip with a
    rail label (Train / Rail_transport / Subway) as positives, plus capped samples
    of known confuser classes (road vehicles, aircraft, engines, weather) and a
    random sample of everything else as negatives.

Writes data/manifest.csv: path,source,id,label,split,labels,license
Splits: ESC-50 folds 1-3 train, 4 val, 5 test. FSD50K dev split train/val, eval -> test.
"""
import csv
import io
import json
import os
import random
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from common import CLIPS, DATA, MANIFEST, decode_bytes, write_wav16

ESC50_URL = "https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip"
FSD_URL = "https://huggingface.co/datasets/Fhrozen/FSD50k/resolve/main/clips/{split}/{fname}.wav"
META = os.path.join(DATA, "meta")

RAIL = {"Train", "Rail_transport", "Subway_and_metro_and_underground"}
# (label, max dev clips, max eval clips)
CONFUSERS = [
    ("Truck", 110, 44), ("Bus", 120, 40), ("Car_passing_by", 126, 60),
    ("Traffic_noise_and_roadway_noise", 152, 60), ("Motorcycle", 100, 40),
    ("Race_car_and_auto_racing", 60, 20), ("Accelerating_and_revving_and_vroom", 80, 30),
    ("Vehicle_horn_and_car_horn_and_honking", 100, 40), ("Car", 120, 40),
    ("Aircraft", 120, 45), ("Fixed-wing_aircraft_and_airplane", 101, 45),
    ("Engine", 150, 50), ("Idling", 80, 30), ("Engine_starting", 40, 15),
    ("Boat_and_Water_vehicle", 80, 30), ("Thunder", 120, 40), ("Thunderstorm", 80, 30),
    ("Wind", 150, 50), ("Rain", 120, 40), ("Waves_and_surf", 60, 20), ("Siren", 80, 30),
    ("Power_tool", 60, 20), ("Drill", 40, 15), ("Mechanical_fan", 60, 20),
    ("Church_bell", 40, 15), ("Explosion", 40, 15), ("Crowd", 60, 20), ("Speech", 80, 30),
    ("Music", 80, 30), ("Walk_and_footsteps", 40, 15), ("Bird", 60, 20), ("Dog", 40, 15),
]
RANDOM_NEG = {"dev": 900, "eval": 250}


def log(msg):
    print(msg, flush=True)


def esc50(rows):
    out_dir = os.path.join(CLIPS, "esc50")
    meta_path = os.path.join(META, "esc50.csv")
    zf = None
    if os.path.exists(meta_path) and os.path.isdir(out_dir) and len(os.listdir(out_dir)) >= 2000:
        log("ESC-50 already present, skipping download")
        meta = list(csv.DictReader(open(meta_path, encoding="utf-8")))
    else:
        log("Downloading ESC-50 zip ...")
        r = requests.get(ESC50_URL, timeout=600)
        r.raise_for_status()
        zf = zipfile.ZipFile(io.BytesIO(r.content))
        meta_name = next(n for n in zf.namelist() if n.endswith("meta/esc50.csv"))
        open(meta_path, "wb").write(zf.read(meta_name))
        meta = list(csv.DictReader(open(meta_path, encoding="utf-8")))
        audio_prefix = meta_name.replace("meta/esc50.csv", "audio/")
    split_of = {1: "train", 2: "train", 3: "train", 4: "val", 5: "test"}
    for i, m in enumerate(meta):
        path = os.path.join(out_dir, m["filename"])
        if not os.path.exists(path):
            write_wav16(path, decode_bytes(zf.read(audio_prefix + m["filename"])))  # zf set when missing
        rows.append({
            "path": os.path.relpath(path, DATA).replace("\\", "/"), "source": "esc50",
            "id": m["filename"], "label": int(m["category"] == "train"),
            "split": split_of[int(m["fold"])], "labels": m["category"],
            "license": "esc50 (CC BY-NC 3.0 dataset; per-clip in ESC-50 LICENSE)",
        })
        if i % 500 == 0:
            log(f"  esc50 {i}/{len(meta)}")
    log(f"ESC-50 done: {sum(r['label'] for r in rows if r['source']=='esc50')} positives")


def fsd_select(seed=1234):
    rnd = random.Random(seed)
    picked = {}
    for split in ("dev", "eval"):
        rows = list(csv.DictReader(open(os.path.join(META, f"{split}.csv"), encoding="utf-8")))
        for r in rows:
            r["labelset"] = set(r["labels"].split(","))
        pos = [r for r in rows if r["labelset"] & RAIL]
        neg_pool = [r for r in rows if not (r["labelset"] & RAIL)]
        for r in pos:
            picked[(split, r["fname"])] = (r, 1)
        col = 1 if split == "dev" else 2
        for conf in CONFUSERS:
            cands = [r for r in neg_pool if conf[0] in r["labelset"]]
            rnd.shuffle(cands)
            for r in cands[:conf[col]]:
                picked.setdefault((split, r["fname"]), (r, 0))
        rnd.shuffle(neg_pool)
        n = 0
        for r in neg_pool:
            if n >= RANDOM_NEG[split]:
                break
            if (split, r["fname"]) not in picked:
                picked[(split, r["fname"])] = (r, 0)
                n += 1
    return picked


def fsd_fetch(split, fname):
    path = os.path.join(CLIPS, "fsd50k", split, f"{fname}.wav")
    if os.path.exists(path):
        return path
    url = FSD_URL.format(split=split, fname=fname)
    last = None
    for attempt in range(8):
        try:
            r = requests.get(url, timeout=120)
            if r.status_code == 429:  # HF anonymous rate limit: back off and retry
                time.sleep(float(r.headers.get("Retry-After", 0)) or min(120, 10 * 2 ** attempt))
                last = "429 Too Many Requests"
                continue
            r.raise_for_status()
            write_wav16(path, decode_bytes(r.content))
            return path
        except Exception as e:  # noqa: BLE001 - retry any network/decode error
            last = e
    raise RuntimeError(f"{url}: {last}")


def fsd50k(rows):
    lic = {}
    for split in ("dev", "eval"):
        info = json.load(open(os.path.join(META, f"{split}_clips_info_FSD50K.json"), encoding="utf-8"))
        lic.update({(split, k): v.get("license", "") for k, v in info.items()})
    picked = fsd_select()
    log(f"FSD50K: {len(picked)} clips selected, "
        f"{sum(1 for v in picked.values() if v[1])} positives")
    done = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(fsd_fetch, s, f): (s, f) for (s, f) in picked}
        for fut in as_completed(futs):
            s, f = futs[fut]
            r, label = picked[(s, f)]
            try:
                path = fut.result()
            except Exception as e:  # noqa: BLE001
                failed += 1
                log(f"  FAIL {e}")
                continue
            split = r["split"] if s == "dev" else "test"
            rows.append({
                "path": os.path.relpath(path, DATA).replace("\\", "/"), "source": "fsd50k",
                "id": f"{s}/{f}", "label": label, "split": split,
                "labels": r["labels"], "license": lic.get((s, f), ""),
            })
            done += 1
            if done % 250 == 0:
                log(f"  fsd50k {done}/{len(picked)} (failed {failed})")
    log(f"FSD50K done: {done} ok, {failed} failed")


def main():
    rows = []
    esc50(rows)
    fsd50k(rows)
    with open(MANIFEST, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "source", "id", "label", "split", "labels", "license"])
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: r["path"]))
    for split in ("train", "val", "test"):
        s = [r for r in rows if r["split"] == split]
        log(f"{split}: {len(s)} clips, {sum(r['label'] for r in s)} positive")
    return 0


if __name__ == "__main__":
    sys.exit(main())
