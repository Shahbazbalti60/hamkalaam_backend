# -*- coding: utf-8 -*-
r"""Is the camera picture mirrored compared with the training data - and does 'try both' fix it?

For the dataset clips the forest has never seen, it tests two situations:
  - the picture is like the training data
  - the picture is mirrored (this dropped the forest to about 31 %)
and three ways of answering: as it is, always flipped, or "try both and take the one the forest is
more sure about" (the app's new automatic way, with a few safety margins).

Run from E:\Projects\hamkalaam_backend (venv active):
    python eval_orientation.py E:\Projects\psl_sequences.npz
"""
import json
import os
import sys

import joblib
import numpy as np
from sklearn.model_selection import train_test_split

from app.ai.forest_classifier import OrientationTracker, mirror_clip
from app.ai.forest_features import clip_features
from app.ai.hand_tidy import tidy_hands

NPZ = sys.argv[1] if len(sys.argv) > 1 else r"E:\Projects\psl_sequences.npz"
MODEL_DIR = os.path.join("app", "ai", "model")
words = json.load(open(os.path.join(MODEL_DIR, "forest_labels.json"), encoding="utf-8"))["words"]
forest = joblib.load(os.path.join(MODEL_DIR, "psl_forest.joblib"))
data = np.load(NPZ, allow_pickle=True)
xk = [k for k in data.files if data[k].ndim == 3 and data[k].shape[-1] == 126]
X = data[xk[0]].astype(np.float32)
N = len(X)
one_d = {k: data[k] for k in data.files if data[k].ndim == 1 and len(data[k]) == N}
text = {k: v for k, v in one_d.items() if v.dtype.kind in "USO"}
ints = {k: v for k, v in one_d.items() if v.dtype.kind in "iu"}
device = next((v for v in text.values() if any("webcam" in str(s) for s in set(v.tolist()))), None)
lab = next((v for v in text.values() if v is not device), None)
y = np.array([words.index(w) for w in lab.tolist()]) if lab is not None else next(iter(ints.values())).astype(int)
_, te = train_test_split(np.arange(N), test_size=0.2, stratify=y, random_state=42)
Xte, yte = X[te], y[te]
cls = np.asarray(forest.classes_, dtype=int)


def probs(clips):
    return forest.predict_proba(np.stack([clip_features(tidy_hands(c)) for c in clips]))


def acc(p, k=1):
    top = cls[np.argsort(p, axis=1)[:, ::-1][:, :k]]
    return float(np.mean([yte[i] in top[i] for i in range(len(yte))])) * 100


P_plain, P_flip = probs(Xte), probs([mirror_clip(c) for c in Xte])           # as recorded / mirrored
print(f"{len(te)} unseen clips\n")
print(f"{'situation':36s} {'answer':34s} {'top-1':>7s} {'top-3':>7s}")
for scene, (as_is, flipped) in {"picture like the training data": (P_plain, P_flip),
                                "picture MIRRORED": (P_flip, P_plain)}.items():
    rows = [("as it is (old behaviour)", as_is), ("always flipped", flipped)]
    for margin in (0.0, 0.05, 0.10):
        pick = flipped.max(1) > as_is.max(1) + margin
        rows.append((f"try both, margin {margin:.2f} (new)", np.where(pick[:, None], flipped, as_is)))
    for name, p in rows:
        print(f"{scene:36s} {name:34s} {acc(p, 1):6.1f}% {acc(p, 3):6.1f}%")
    print()

# ---- the policy the app now uses: it learns the orientation from a stream of clips ----
print("The app's automatic policy, simulated on a stream of the unseen clips in random order")
print("(the first clips are answered by 'try both', then the orientation locks in):")
rng = np.random.default_rng(3)
for scene, (as_is, flipped) in {"picture like the training data": (P_plain, P_flip),
                                "picture MIRRORED": (P_flip, P_plain)}.items():
    hits1 = hits3 = 0
    tr = OrientationTracker()
    for i in rng.permutation(len(yte)):
        a, b = as_is[i], flipped[i]
        use_flip = tr.decide(a.max(), b.max()); tr.learn(a.max(), b.max())
        p = b if use_flip else a
        top = cls[np.argsort(p)[::-1][:3]]
        hits1 += top[0] == yte[i]; hits3 += yte[i] in top
    print(f"  {scene:36s} top-1 {hits1/len(yte)*100:5.1f}%   top-3 {hits3/len(yte)*100:5.1f}%")
