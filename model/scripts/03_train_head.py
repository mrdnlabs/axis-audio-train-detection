"""Train the binary train / not-train head and evaluate it.

Head input = YAMNet 1024-d embedding ++ YAMNet's 521 class scores (recomputed from the
embedding with YAMNet's own classifier layer), so the head can use explicit evidence such
as "Bus", "Truck" or "Aircraft" against the rail classes.

After training, the output bias is shifted so that the clip-level F1-optimal threshold on
the validation set maps to 0.5 (= Sensitivity ~50 on the device).

Writes out/head.h5 and out/metrics.json.
"""
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "yamnet"))
from common import DATA, MANIFEST, OUT  # noqa: E402

SILENCE_DB = -70.0       # windows quieter than this are dropped (no information)
POS_QUIET_MARGIN = 25.0  # positive-clip windows >25 dB below the clip's loudest are dropped
HARD_NEG_WEIGHT = 2.0
HARD_NEG_LABELS = {
    "Vehicle", "Motor_vehicle_(road)", "Truck", "Bus", "Car", "Car_passing_by", "Motorcycle",
    "Traffic_noise_and_roadway_noise", "Race_car_and_auto_racing", "Accelerating_and_revving_and_vroom",
    "Vehicle_horn_and_car_horn_and_honking", "Aircraft", "Fixed-wing_aircraft_and_airplane", "Engine",
    "Idling", "Boat_and_Water_vehicle", "Thunder", "Thunderstorm", "Wind", "Siren",
    # ESC-50 categories
    "airplane", "engine", "helicopter", "car_horn", "thunderstorm", "wind", "siren", "church_bells",
}
HOP_S = 0.48


def yamnet_classifier():
    """YAMNet's final Dense layer (embedding -> 521 logits)."""
    import params as yamnet_params
    import yamnet as yamnet_lib
    from tf_keras import layers
    m = yamnet_lib.yamnet_frames_model(yamnet_params.Params())
    m.load_weights(os.path.join(DATA, "yamnet.h5"))
    dense = [l for l in m.layers if isinstance(l, layers.Dense)][-1]
    w, b = dense.get_weights()
    return w.astype(np.float32), b.astype(np.float32)


def features(emb, w, b):
    emb = emb.astype(np.float32)
    scores = 1.0 / (1.0 + np.exp(-(emb @ w + b)))
    return np.concatenate([emb, scores.astype(np.float32)], axis=1)


def load():
    d = dict(np.load(os.path.join(OUT, "embeddings.npz")))
    rms = d["rms_db"].astype(np.float32)
    keep = rms > SILENCE_DB
    # Weak (clip-level) labels: in long positive clips the quiet lead-in/out often
    # has no train in it. Drop windows far below the clip's loudest window.
    pos = d["label"] == 1
    for c in np.unique(d["clip"][pos]):
        for a in np.unique(d["aug"][d["clip"] == c]):
            m = (d["clip"] == c) & (d["aug"] == a)
            keep[m] &= rms[m] > rms[m].max() - POS_QUIET_MARGIN
    return {k: v[keep] for k, v in d.items()}


def build_head(dim):
    from tf_keras import Sequential, layers, regularizers
    return Sequential([
        layers.Input(shape=(dim,)),
        layers.Dropout(0.2),
        layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(1e-4), name="head_dense"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid", name="head_out"),
    ], name="train_head")


def smooth(s, n=4):
    return np.convolve(s, np.ones(n) / n, mode="valid") if len(s) >= n else s


def clip_scores(score, clip):
    """Clip score = max over a 4-window (~2 s) moving average, like the device smoothing."""
    return {c: float(smooth(score[clip == c]).max()) for c in np.unique(clip)}


def event_detected(s, thr, trigger_windows):
    """Device-like rule: smoothed score >= thr for `trigger_windows` consecutive windows."""
    run = 0
    for v in smooth(s):
        run = run + 1 if v >= thr else 0
        if run >= trigger_windows:
            return True
    return False


def prf(y, yhat):
    tp = int(np.sum(yhat & (y == 1)))
    fp = int(np.sum(yhat & (y == 0)))
    fn = int(np.sum(~yhat & (y == 1)))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4), "tp": tp, "fp": fp, "fn": fn}


def evaluate(name, y, p, clip, rows):
    from sklearn.metrics import average_precision_score, roc_auc_score
    res = {"patch_auc": round(float(roc_auc_score(y, p)), 4),
           "patch_ap": round(float(average_precision_score(y, p)), 4)}
    cs = clip_scores(p, clip)
    cy = np.array([int(rows[c]["label"]) for c in cs])
    cp = np.array(list(cs.values()))
    res["clip_auc"] = round(float(roc_auc_score(cy, cp)), 4)
    res["clip_ap"] = round(float(average_precision_score(cy, cp)), 4)
    res["clip_at"] = {str(t): prf(cy, cp >= t) for t in (0.3, 0.5, 0.7, 0.85)}
    # Device-like event rule at threshold 0.5 with 1.5 s and 3 s trigger durations.
    for trig_s in (1.5, 3.0):
        k = max(1, int(round(trig_s / HOP_S)))
        det = np.array([event_detected(p[clip == c], 0.5, k) for c in cs])
        res[f"event@0.5_trigger{trig_s}s"] = prf(cy, det)
    fpr = {}
    for c in cs:
        if rows[c]["label"] != "0":
            continue
        for lab in set(rows[c]["labels"].split(",")) & HARD_NEG_LABELS:
            fpr.setdefault(lab, []).append(cs[c] >= 0.5)
    res["confuser_fp@0.5"] = {k: {"n": len(v), "fp_rate": round(float(np.mean(v)), 3)}
                              for k, v in sorted(fpr.items()) if len(v) >= 8}
    print(f"== {name}: auc {res['clip_auc']} ap {res['clip_ap']} | clip@0.5 {res['clip_at']['0.5']} | "
          f"event3s {res['event@0.5_trigger3.0s']}", flush=True)
    return res


def main():
    import tensorflow as tf
    from tf_keras import callbacks, metrics as kmetrics, optimizers
    tf.random.set_seed(1)
    rows = list(csv.DictReader(open(MANIFEST, encoding="utf-8")))
    d = load()
    w, b = yamnet_classifier()
    x = features(d["emb"], w, b)
    y = d["label"].astype(np.float32)
    tr, va, te = (d["split"] == 0), (d["split"] == 1), (d["split"] == 2)
    print(f"patches train {tr.sum()} ({int(y[tr].sum())} pos)  val {va.sum()} ({int(y[va].sum())} pos)  "
          f"test {te.sum()} ({int(y[te].sum())} pos); features {x.shape[1]}", flush=True)

    hard = np.array([bool(set(rows[c]["labels"].split(",")) & HARD_NEG_LABELS) for c in d["clip"]])
    sw = np.where((y == 0) & hard, HARD_NEG_WEIGHT, 1.0).astype(np.float32)

    head = build_head(x.shape[1])
    head.compile(optimizer=optimizers.Adam(1e-3), loss="binary_crossentropy",
                 metrics=[kmetrics.AUC(name="auc", curve="PR")])
    head.fit(x[tr], y[tr], sample_weight=sw[tr], validation_data=(x[va], y[va]), epochs=60, batch_size=512,
             verbose=2, callbacks=[callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=8,
                                                           restore_best_weights=True)])

    # Calibrate: move the val clip-level F1-optimal threshold to 0.5 by shifting the output bias.
    orig = d["aug"] == 0
    mv = va & orig
    pv = head.predict(x[mv], batch_size=4096, verbose=0).ravel()
    cs = clip_scores(pv, d["clip"][mv])
    cy = np.array([int(rows[c]["label"]) for c in cs])
    cp = np.array(list(cs.values()))
    grid = np.linspace(0.05, 0.98, 94)
    f1s = [prf(cy, cp >= t)["f1"] for t in grid]
    t_star = float(grid[int(np.argmax(f1s))])
    out_layer = head.get_layer("head_out")
    kw, kb = out_layer.get_weights()
    out_layer.set_weights([kw, kb - np.log(t_star / (1 - t_star))])
    print(f"calibration: val F1-optimal clip threshold {t_star:.2f} (F1 {max(f1s):.3f}) -> mapped to 0.5", flush=True)
    head.save(os.path.join(OUT, "head.h5"))

    results = {"counts": {"train_patches": int(tr.sum()), "val_patches": int(va.sum()), "test_patches": int(te.sum())},
               "head_input": "embedding(1024) ++ yamnet_scores(521)", "calibration_threshold": t_star}
    for name, m in (("val", va & orig), ("test", te & orig)):
        p = head.predict(x[m], batch_size=4096, verbose=0).ravel()
        results[f"head_{name}"] = evaluate(f"head {name}", y[m], p, d["clip"][m], rows)
        results[f"yamnet_rail_{name}"] = evaluate(f"yamnet-rail {name}", y[m], d["rail"][m].astype(np.float32),
                                                  d["clip"][m], rows)
    json.dump(results, open(os.path.join(OUT, "metrics.json"), "w"), indent=2)
    print("saved head.h5 + metrics.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
