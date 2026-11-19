# -*- coding: utf-8 -*-
r"""Why is live recognition worse than the dataset test? This script finds out - and which fix helps.

It takes the dataset clips the forest has NEVER seen (the same 20 % test split as the training
notebook) and cuts them like the live app does (only the part where a hand is visible). Then it
compares three ways of feeding such a cut word to the forest, also for faster and slower signing:

  A  stretch   - the cut word is stretched to 60 frames        (what the app does now)
  B  natural   - the word keeps its real speed, empty frames fill the rest of the 60
  C  multi     - the word is tried at several speeds and the answers are averaged

Run from E:\Projects\hamkalaam_backend (venv active):
    python eval_segment_strategies.py E:\Projects\psl_sequences.npz
"""
import json
import os
import sys
import time

import joblib
import numpy as np
from sklearn.model_selection import train_test_split

from app.ai.forest_features import clip_features
from app.ai.hand_tidy import tidy_hands

NPZ = sys.argv[1] if len(sys.argv) > 1 else r"E:\Projects\psl_sequences.npz"
MODEL_DIR = os.path.join("app", "ai", "model")
PRE_POST = 9                      # frames kept before / after the hand (the app keeps 0.3 s)
FRACTIONS = [0.55, 0.70, 0.85, 1.00]
SPEEDS = [("faster signing (x0.7)", 0.7), ("same speed", 1.0), ("slower signing (x1.4)", 1.4)]

# ---------------------------------------------------------------- load
words = json.load(open(os.path.join(MODEL_DIR, "forest_labels.json"), encoding="utf-8"))["words"]
forest = joblib.load(os.path.join(MODEL_DIR, "psl_forest.joblib"))
data = np.load(NPZ, allow_pickle=True)
x_keys = [k for k in data.files if data[k].ndim == 3 and data[k].shape[-1] == 126]
if len(x_keys) != 1:
    sys.exit(f"Could not find the (clips, 60, 126) array. Keys: {data.files}")
X = data[x_keys[0]].astype(np.float32)
N = len(X)
one_d = {k: data[k] for k in data.files if data[k].ndim == 1 and len(data[k]) == N}
text = {k: v for k, v in one_d.items() if v.dtype.kind in "USO"}
ints = {k: v for k, v in one_d.items() if v.dtype.kind in "iu"}
device = next((v for v in text.values() if any("webcam" in str(s) for s in set(v.tolist()))), None)
label_txt = next((v for v in text.values() if v is not device), None)
if label_txt is not None:
    y = np.array([words.index(w) for w in label_txt.tolist()])
elif ints:
    y = next(iter(ints.values())).astype(int)
else:
    sys.exit(f"Could not find the labels. Arrays: { {k: v.shape for k, v in one_d.items()} }")

# the SAME split as the training notebook (random_state 42, stratified, 20 % test)
_, te = train_test_split(np.arange(N), test_size=0.2, stratify=y, random_state=42)
print(f"{N} clips, test clips the forest never saw: {len(te)}")


# ---------------------------------------------------------------- helpers
def hand_span(clip):
    present = np.abs(clip).sum(axis=1) > 0
    idx = np.where(present)[0]
    if len(idx) == 0:
        return None
    a, b = max(0, idx[0] - PRE_POST), min(len(clip), idx[-1] + 1 + PRE_POST)
    return clip[a:b]


def resample(seg, n):
    n = max(2, int(n))
    idx = np.round(np.linspace(0, len(seg) - 1, n)).astype(int)
    return seg[idx]


def pad_center(seg):
    out = np.zeros((60, 126), np.float32)
    n = min(len(seg), 60)
    off = (60 - n) // 2
    out[off:off + n] = seg[:n]
    return out


def feats(arr60):
    return clip_features(tidy_hands(arr60))


def to_rows(arrays):
    return np.stack([feats(a) for a in arrays])


def topk(probs, truth, k):
    cls = np.asarray(forest.classes_, dtype=int)
    top = cls[np.argsort(probs, axis=1)[:, ::-1][:, :k]]
    return float(np.mean([truth[i] in top[i] for i in range(len(truth))]))


# ---------------------------------------------------------------- baseline + span statistics
spans, truth = [], []
for i in te:
    sp = hand_span(X[i])
    if sp is not None:
        spans.append(sp); truth.append(y[i])
truth = np.array(truth)
lens = np.array([len(s) for s in spans])
lead = np.mean([np.argmax(np.abs(X[i]).sum(1) > 0) for i in te])
print(f"hand-visible part of a clip: median {np.median(lens):.0f} frames, "
      f"{(lens < 60).mean() * 100:.0f} % of clips are shorter than 60 frames "
      f"(average {lead:.0f} empty frames at the start of a dataset clip)")

t0 = time.time()
base = forest.predict_proba(to_rows([X[i] for i in te]))
print(f"baseline (full dataset clips as in training): top-1 {topk(base, y[te], 1)*100:.1f} %  "
      f"top-3 {topk(base, y[te], 3)*100:.1f} %   <- should be about the forest_main_test_acc of the notebook\n")

# ---------------------------------------------------------------- the three strategies
print(f"{'':26s} {'A stretch (now)':>18s} {'B natural speed':>18s} {'C several speeds':>18s}")
print(f"{'':26s} {'top1   top3':>18s} {'top1   top3':>18s} {'top1   top3':>18s}")
for name, factor in SPEEDS:
    segs = [resample(s, len(s) * factor) if factor != 1.0 else s for s in spans]
    pA = forest.predict_proba(to_rows([resample(s, 60) for s in segs]))
    pB = forest.predict_proba(to_rows([pad_center(s) if len(s) < 60 else resample(s, 60) for s in segs]))
    pC = np.mean([forest.predict_proba(to_rows([pad_center(resample(s, 60 * f)) for s in segs]))
                  for f in FRACTIONS], axis=0)
    row = f"{name:26s}"
    for p in (pA, pB, pC):
        row += f" {topk(p, truth, 1)*100:8.1f} {topk(p, truth, 3)*100:7.1f}  "
    print(row)
print(f"\n(took {time.time() - t0:.0f} s)  Send me this whole table.")
