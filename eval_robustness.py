# -*- coding: utf-8 -*-
r"""Which real-life difference hurts the Random Forest?

Takes the dataset clips the forest has NEVER seen and changes them the way a live camera could
differ from the training recordings: mirrored picture, hands swapped, person standing somewhere else
in the picture, hand smaller, different depth (z) values, noisy landmarks, missing frames.
Prints the accuracy for each change. A big drop = that is a weakness of the model.

Run from E:\Projects\hamkalaam_backend (venv active):
    python eval_robustness.py E:\Projects\psl_sequences.npz
"""
import json
import os
import sys

import joblib
import numpy as np
from sklearn.model_selection import train_test_split

from app.ai.forest_features import clip_features
from app.ai.hand_tidy import tidy_hands

NPZ = sys.argv[1] if len(sys.argv) > 1 else r"E:\Projects\psl_sequences.npz"
MODEL_DIR = os.path.join("app", "ai", "model")
rng = np.random.default_rng(0)

words = json.load(open(os.path.join(MODEL_DIR, "forest_labels.json"), encoding="utf-8"))["words"]
forest = joblib.load(os.path.join(MODEL_DIR, "psl_forest.joblib"))
data = np.load(NPZ, allow_pickle=True)
x_keys = [k for k in data.files if data[k].ndim == 3 and data[k].shape[-1] == 126]
X = data[x_keys[0]].astype(np.float32)
N = len(X)
one_d = {k: data[k] for k in data.files if data[k].ndim == 1 and len(data[k]) == N}
text = {k: v for k, v in one_d.items() if v.dtype.kind in "USO"}
ints = {k: v for k, v in one_d.items() if v.dtype.kind in "iu"}
device = next((v for v in text.values() if any("webcam" in str(s) for s in set(v.tolist()))), None)
label_txt = next((v for v in text.values() if v is not device), None)
y = np.array([words.index(w) for w in label_txt.tolist()]) if label_txt is not None else next(iter(ints.values())).astype(int)
_, te = train_test_split(np.arange(N), test_size=0.2, stratify=y, random_state=42)
Xte, yte = X[te], y[te]
print(f"{len(te)} unseen clips\n")


def hands_view(a):
    h = a.reshape(60, 2, 21, 3).copy()
    return h, (np.abs(h).sum(-1) > 0)           # (60,2,21,3), (60,2,21) where a landmark exists


def back(h):
    return h.reshape(60, 126)


def mirror(a):
    h, nz = hands_view(a); h[..., 0] = np.where(nz, 1 - h[..., 0], 0); return back(h)

def swap_hands(a):
    h, _ = hands_view(a); return back(h[:, ::-1])

def shift(dx, dy):
    def f(a):
        h, nz = hands_view(a); h[..., 0] += np.where(nz, dx, 0); h[..., 1] += np.where(nz, dy, 0); return back(h)
    return f

def smaller_in_frame(s):                          # whole picture content moves towards the centre (person farther away)
    def f(a):
        h, nz = hands_view(a)
        for c in (0, 1):
            h[..., c] = np.where(nz, 0.5 + (h[..., c] - 0.5) * s, 0)
        return back(h)
    return f

def z_scale(k):
    def f(a):
        h, nz = hands_view(a); h[..., 2] = np.where(nz, h[..., 2] * k, 0); return back(h)
    return f

def noise(sig):
    def f(a):
        h, nz = hands_view(a); h += (rng.normal(0, sig, h.shape) * nz[..., None]).astype(np.float32); return back(h)
    return f

def drop_frames(frac):
    def f(a):
        a = a.copy(); a[rng.random(60) < frac] = 0; return a
    return f

TESTS = [
    ("no change (reference)", lambda a: a),
    ("picture mirrored", mirror),
    ("the two hand slots swapped", swap_hands),
    ("hands 0.15 to the right", shift(0.15, 0.0)),
    ("hands 0.15 lower in the picture", shift(0.0, 0.15)),
    ("person farther away (x0.7 around the centre)", smaller_in_frame(0.7)),
    ("depth (z) values x0.5", z_scale(0.5)),
    ("depth (z) values x2", z_scale(2.0)),
    ("landmark noise 0.004", noise(0.004)),
    ("landmark noise 0.010", noise(0.010)),
    ("25 % of the frames missing", drop_frames(0.25)),
]


def topk(p, truth, k):
    cls = np.asarray(forest.classes_, dtype=int)
    top = cls[np.argsort(p, axis=1)[:, ::-1][:, :k]]
    return float(np.mean([truth[i] in top[i] for i in range(len(truth))]))


print(f"{'change':48s} {'top-1':>7s} {'top-3':>7s}")
for name, fn in TESTS:
    F = np.stack([clip_features(tidy_hands(fn(c))) for c in Xte])
    p = forest.predict_proba(F)
    print(f"{name:48s} {topk(p, yte, 1)*100:6.1f}% {topk(p, yte, 3)*100:6.1f}%")

# which kinds of numbers does the forest rely on? (assumes the tidy columns are x,y,z per landmark,
# then wrist position, then the hand-present flags)
imp = forest.feature_importances_.reshape(8, 132).sum(0)
share = {
    "finger positions x": imp[0:126][0::3].sum(), "finger positions y": imp[0:126][1::3].sum(),
    "finger depth z": imp[0:126][2::3].sum(), "where the hand is in the picture (wrist)": imp[126:130].sum(),
    "hand-present flags": imp[130:132].sum(),
}
print("\nWhat the forest relies on (share of its decisions):")
for k, v in share.items():
    print(f"  {k:42s} {v*100:5.1f} %")
