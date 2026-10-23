"""
test_sequence_api.py  -  checks the REAL model through your REAL running server.

Run it from the backend folder, with the backend venv active and Flask running:
    python test_sequence_api.py
It takes random recordings from psl_sequences.npz, sends each one to
/api/recognition/classify-sequence exactly like the app will, and prints how
many came back right. (Its numbers should look like the Colab results.)

Optional arguments:
    python test_sequence_api.py <path to psl_sequences.npz> <how many recordings>
"""
import sys
import time

import numpy as np
import requests

BASE = "http://127.0.0.1:5000/api"
NPZ = sys.argv[1] if len(sys.argv) > 1 else r"E:\Projects\psl_sequences.npz"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 60

z = np.load(NPZ)
X, y, device = z["X"], z["y"].astype(int), z["device"].astype(int)
words = [str(w) for w in z["words"]]
devices = [str(d) for d in z["devices"]]

r = requests.post(f"{BASE}/auth/login",
                  json={"email": "admintest@hamkalaam.com", "password": "Test1234"}, timeout=20)
r.raise_for_status()
headers = {"Authorization": "Bearer " + r.json()["access_token"]}

t0 = time.time()
r = requests.get(f"{BASE}/recognition/model-status", headers=headers, timeout=180)
print(f"model-status: {r.json()}   (first call loads the model: {time.time() - t0:.1f}s)")
if not r.json().get("ready"):
    sys.exit("The model is not ready - read the reason above.")

rng = np.random.RandomState(1)
chosen = rng.choice(len(X), min(N, len(X)), replace=False)
right = top3 = 0
per_device = {d: [0, 0] for d in devices}
times, mistakes = [], []
for i in chosen:
    t = time.time()
    resp = requests.post(f"{BASE}/recognition/classify-sequence",
                         json={"frames": X[i].tolist()}, headers=headers, timeout=60)
    times.append(time.time() - t)
    resp.raise_for_status()
    out = resp.json()
    truth = words[y[i]]
    ok = out["word"] == truth
    right += ok
    top3 += truth in [c["word"] for c in out["top3"]]
    per_device[devices[device[i]]][0] += ok
    per_device[devices[device[i]]][1] += 1
    if not ok:
        mistakes.append((truth, out["word"], out["confidence"]))

n = len(chosen)
print(f"\nrecordings sent: {n}")
print(f"correct word   : {right}/{n}  ({right / n * 100:.1f}%)")
print(f"in top 3       : {top3}/{n}  ({top3 / n * 100:.1f}%)")
for d, (a, b) in per_device.items():
    if b:
        print(f"   {d:12s}: {a}/{b}")
print(f"time per request: {np.mean(times) * 1000:.0f} ms average, {np.max(times) * 1000:.0f} ms slowest")
if mistakes:
    print("mistakes (true -> predicted, confidence):")
    for t_, p_, c_ in mistakes[:10]:
        print(f"   {t_} -> {p_}   {c_:.2f}")
print("\nDone. Paste the lines above back to Claude.")
