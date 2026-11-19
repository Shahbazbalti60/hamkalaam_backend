# -*- coding: utf-8 -*-
r"""Are the live camera numbers different from the training numbers?

1. Put   LOG_LIVE_SAMPLES=1   in the backend .env and restart the backend.
2. Sign about 15 words in the app (the clips are saved in uploads\live_samples\samples.jsonl).
3. Run from E:\Projects\hamkalaam_backend (venv active):
       python analyze_live_samples.py E:\Projects\psl_sequences.npz
"""
import json
import os
import sys
from collections import Counter

import numpy as np

NPZ = sys.argv[1] if len(sys.argv) > 1 else r"E:\Projects\psl_sequences.npz"
LIVE = os.path.join("uploads", "live_samples", "samples.jsonl")
if not os.path.exists(LIVE):
    sys.exit("No live samples yet. Add LOG_LIVE_SAMPLES=1 to .env, restart the backend and sign some words.")

live, meta = [], []
for line in open(LIVE, encoding="utf-8"):
    try:
        r = json.loads(line)
        live.append(np.array(r["frames"], dtype=np.float32)); meta.append(r)
    except Exception:
        pass
print(f"{len(live)} live clips\n")

data = np.load(NPZ, allow_pickle=True)
xk = [k for k in data.files if data[k].ndim == 3 and data[k].shape[-1] == 126][0]
X = data[xk].astype(np.float32)
sel = np.random.default_rng(1).choice(len(X), size=min(800, len(X)), replace=False)


def stats(clip):
    h = clip.reshape(60, 2, 21, 3)
    present = (np.abs(h).sum(-1) > 0)                       # (60,2,21)
    hand = present.any(-1)                                    # (60,2)
    out = {"frames with a hand (%)": hand.any(1).mean() * 100,
           "frames with BOTH hands (%)": hand.all(1).mean() * 100,
           "share of hand frames in slot 0 (%)": (hand[:, 0].sum() / max(1, hand.sum())) * 100}
    sizes, zr, wx, wy = [], [], [], []
    for t in range(60):
        for k in range(2):
            if hand[t, k]:
                p = h[t, k]
                sizes.append(float(np.linalg.norm(p[9, :2] - p[0, :2])))      # wrist -> middle knuckle
                zr.append(float(np.ptp(p[:, 2])))
                wx.append(float(p[0, 0])); wy.append(float(p[0, 1]))
    if sizes:
        out["hand size in the picture (wrist to knuckle)"] = float(np.median(sizes))
        out["depth range of a hand (z)"] = float(np.median(zr))
        out["depth range / hand size"] = float(np.median(zr) / max(np.median(sizes), 1e-6))
        out["wrist position x (0 left .. 1 right)"] = float(np.median(wx))
        out["wrist position y (0 top .. 1 bottom)"] = float(np.median(wy))
    return out


def table(clips):
    rows = [stats(c) for c in clips]
    keys = sorted({k for r in rows for k in r})
    return {k: np.array([r[k] for r in rows if k in r]) for k in keys}


D, L = table(X[sel]), table(live)
print(f"{'':46s} {'TRAINING data':>22s} {'LIVE':>10s}")
print(f"{'':46s} {'median  (10%..90%)':>22s} {'median':>10s}")
for k in D:
    if k not in L:
        continue
    lo, med, hi = np.percentile(D[k], [10, 50, 90]); lv = np.median(L[k])
    flag = "   <-- DIFFERENT" if (lv < lo or lv > hi) else ""
    print(f"{k:46s} {med:8.3f} ({lo:6.3f}..{hi:6.3f}) {lv:10.3f}{flag}")

words = Counter(m["word"] for m in meta)
print("\nwhat the model said:", dict(words.most_common(12)))
print("average confidence: %.2f" % np.mean([m["confidence"] for m in meta]))
print("\nSend me this whole output.")
