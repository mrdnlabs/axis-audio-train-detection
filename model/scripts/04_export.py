"""Export the device model: waveform (1, 15600) float32 @16 kHz -> [train_prob (1,1), yamnet_scores (1,521)].

The STFT uses YAMNet's TFLite-compatible DFT-matrix path, so the TFLite graph only
contains builtin ops and the device feeds raw audio (no feature code in C).

Also writes test vectors for on-device parity checks:
  out/testvec/<name>.wav  (16 kHz PCM16) and out/testvec/expected.json
with per-window outputs using the device's windowing (15600 window, 7680 hop).
"""
import csv
import json
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "yamnet"))
import features as features_lib  # noqa: E402
import params as yamnet_params  # noqa: E402
import yamnet as yamnet_lib  # noqa: E402

from common import DATA, MANIFEST, OUT, YAMNET_HORN, YAMNET_RAIL, read_wav16, write_wav16  # noqa: E402

PATCH = 15600
HOP = 7680


def build(tf):
    from tf_keras import Model, layers, models
    params = yamnet_params.Params(tflite_compatible=True)
    wave_in = layers.Input(batch_shape=(1, PATCH), dtype=tf.float32, name="waveform")
    wave = layers.Lambda(lambda w: tf.reshape(w, [PATCH]))(wave_in)
    _, patches = features_lib.waveform_to_log_mel_spectrogram_patches(wave, params)
    scores, emb = yamnet_lib.yamnet(patches, params)
    backbone = Model(wave_in, [scores, emb], name="yamnet_fixed")
    backbone.load_weights(os.path.join(DATA, "yamnet.h5"))
    head = models.load_model(os.path.join(OUT, "head.h5"), compile=False)
    prob = head(layers.Concatenate(axis=-1)([emb, scores]))  # same features as 03_train_head.py
    return Model(wave_in, [prob, scores], name="traindet")


def windows(wave):
    if len(wave) < PATCH:
        wave = np.pad(wave, (0, PATCH - len(wave)))
    n = 1 + (len(wave) - PATCH) // HOP
    return [wave[i * HOP:i * HOP + PATCH] for i in range(n)]


def main():
    import tensorflow as tf
    model = build(tf)

    @tf.function(input_signature=[tf.TensorSpec([1, PATCH], tf.float32)])
    def serve(waveform):
        prob, scores = model(waveform)
        return {"a_train_prob": prob, "b_yamnet_scores": scores}

    conv = tf.lite.TFLiteConverter.from_concrete_functions([serve.get_concrete_function()], model)
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    tfl = conv.convert()
    path = os.path.join(OUT, "traindet.tflite")
    open(path, "wb").write(tfl)
    print(f"wrote {path} ({len(tfl)/1e6:.1f} MB)")

    interp = tf.lite.Interpreter(model_path=path)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    outs = interp.get_output_details()
    print("input", inp["shape"], inp["dtype"])
    for o in outs:
        print("output", o["index"], o["name"], o["shape"])
    ops = sorted({op["op_name"] for op in interp._get_ops_details()})
    print("ops:", ops)
    prob_idx = next(o["index"] for o in outs if o["shape"][-1] == 1)
    score_idx = next(o["index"] for o in outs if o["shape"][-1] == 521)

    def run_tfl(w):
        interp.set_tensor(inp["index"], w.reshape(1, PATCH).astype(np.float32))
        interp.invoke()
        return float(interp.get_tensor(prob_idx).ravel()[0]), interp.get_tensor(score_idx).ravel()

    # Pick test vectors: 3 positives + 3 hard negatives from the held-out test split.
    rows = list(csv.DictReader(open(MANIFEST, encoding="utf-8")))
    test = [r for r in rows if r["split"] == "test"]
    rnd = np.random.default_rng(3)
    pick = []
    pos = [r for r in test if r["label"] == "1"]
    pick += [pos[i] for i in rnd.choice(len(pos), 3, replace=False)]
    for lab in ("Truck", "Aircraft", "Thunder"):
        c = [r for r in test if r["label"] == "0" and lab in r["labels"]]
        if c:
            pick.append(c[rnd.integers(len(c))])
    tv_dir = os.path.join(OUT, "testvec")
    shutil.rmtree(tv_dir, ignore_errors=True)
    os.makedirs(tv_dir)
    expected = {}
    max_diff = 0.0
    for r in pick:
        wave = read_wav16(os.path.join(DATA, r["path"]))[:16000 * 20]
        name = ("pos_" if r["label"] == "1" else "neg_") + r["id"].replace("/", "_").replace(".wav", "")
        write_wav16(os.path.join(tv_dir, name + ".wav"), wave)
        wave = read_wav16(os.path.join(tv_dir, name + ".wav"))  # exactly what the device will get
        res = []
        for w in windows(wave):
            p, s = run_tfl(w)
            pk, sk = model(w.reshape(1, PATCH))
            max_diff = max(max_diff, abs(p - float(pk.numpy().ravel()[0])),
                           float(np.max(np.abs(s - sk.numpy().ravel()))))
            res.append({"prob": round(p, 5), "rail": round(float(s[YAMNET_RAIL].max()), 5),
                        "horn": round(float(s[YAMNET_HORN].max()), 5), "top": int(np.argmax(s))})
        expected[name] = {"label": int(r["label"]), "labels": r["labels"], "windows": res}
        print(f"{name}: label={r['label']} max_prob={max(x['prob'] for x in res):.3f} "
              f"mean_prob={np.mean([x['prob'] for x in res]):.3f}")
    json.dump(expected, open(os.path.join(tv_dir, "expected.json"), "w"), indent=1)

    # End-to-end clip for the live inject test: the held-out train recording (<= 30 s)
    # with the longest run of smoothed (4-window) score >= 0.6.
    best = (0, None, None)
    for r in pos:
        wave = read_wav16(os.path.join(DATA, r["path"]))[:16000 * 30]
        if len(wave) < 16000 * 8:
            continue
        probs = np.array([run_tfl(w)[0] for w in windows(wave)])
        sm = np.convolve(probs, np.ones(4) / 4, mode="valid") if len(probs) >= 4 else probs
        run = longest = 0
        for v in sm:
            run = run + 1 if v >= 0.6 else 0
            longest = max(longest, run)
        if longest > best[0]:
            best = (longest, r, wave)
    if best[1] is not None:
        write_wav16(os.path.join(tv_dir, "e2e_train.wav"), best[2])
        print(f"e2e clip: {best[1]['id']} ({len(best[2]) / 16000:.1f} s, {best[0]} windows >= 0.6 smoothed)")
    print(f"Keras vs TFLite max abs diff: {max_diff:.2e}")
    info = {"input": [1, PATCH], "sample_rate": 16000, "hop_samples": HOP,
            "outputs": {"train_prob": int(prob_idx), "yamnet_scores": int(score_idx)},
            "ops": ops, "keras_tflite_max_abs_diff": max_diff}
    json.dump(info, open(os.path.join(OUT, "model_info.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
