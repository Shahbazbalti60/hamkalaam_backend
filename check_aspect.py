import json
import os
import sys

import numpy as np

NPZ = sys.argv[1] if len(sys.argv) > 1 else r"E:\Projects\psl_sequences.npz"
LIVE = os.path.join("uploads", "live_samples", "samples.jsonl")


def width_ratio(clips):
    sx, sy = [], []
    for c in clips:
        h = np.asarray(c, dtype=np.float32).reshape(-1, 2, 21, 3)
        present = np.abs(h).sum(-1) > 0
        for t in range(h.shape[0]):
            for k in range(2):
                if present[t, k].all():
                    p = h[t, k]
                    sx.append(float(p[:, 0].std()))
                    sy.append(float(p[:, 1].std()))
    if not sx:
        return None, 0
    return float(np.median(sx) / max(np.median(sy), 1e-6)), len(sx)


data = np.load(NPZ, allow_pickle=True)
xk = [k for k in data.files if data[k].ndim == 3 and data[k].shape[-1] == 126][0]
X = data[xk].astype(np.float32)
N = len(X)
rng = np.random.default_rng(1)

print(f"{'group':34s} {'hand width/height':>18s} {'hand frames':>12s}")
dev = None
for k in data.files:
    v = data[k]
    if v.ndim == 1 and len(v) == N and v.dtype.kind in "USO":
        if any("webcam" in str(s) for s in set(v.tolist())):
            dev = np.array([str(s).replace("\\","/").split("/")[0] for s in v.tolist()])
if dev is not None:
    for d in sorted(set(dev.tolist())):
        idx = np.where(dev == d)[0]
        idx = rng.choice(idx, size=min(300, len(idx)), replace=False)
        r, n = width_ratio(X[idx])
        print(f"training: {d:24s} {r if r is None else round(r, 3)!s:>18s} {n:12d}")
idx = rng.choice(N, size=min(600, N), replace=False)
r, n = width_ratio(X[idx])
print(f"{'training: all':34s} {round(r, 3)!s:>18s} {n:12d}")

if os.path.exists(LIVE):
    live = []
    for line in open(LIVE, encoding="utf-8"):
        try:
            live.append(np.array(json.loads(line)["frames"], dtype=np.float32))
        except Exception:
            pass
    r, n = width_ratio(live)
    print(f"{'LIVE phone (' + str(len(live)) + ' clips)':34s} {r if r is None else round(r, 3)!s:>18s} {n:12d}")
else:
    print("no live samples found")
print("Send me this whole output.")

