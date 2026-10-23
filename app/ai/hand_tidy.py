"""
hand_tidy.py  -  ONE shared "tidy the hands" step.

The SAME file is used (1) when training in Google Colab and (2) later
inside the Flask backend, so training data and live camera data are
always prepared in exactly the same way.

Input : raw landmarks, shape (T, 126) or (N, T, 126)
        = 2 blocks of 63 numbers (21 points x x,y,z), in whatever slot order
Output: features,       shape (T, 132) or (N, T, 132)

What it does, per frame:
  1. Puts the hand(s) in a FIXED order: a single hand always goes in
     block 0; with two hands the one further LEFT on the screen goes first.
     -> it no longer matters which slot the recorder used for which hand.
  2. Makes each hand relative to its own wrist and scales it by its own
     size -> less dependent on camera distance and device.
  3. Adds the wrist position (x,y) so "where the hand is" is not lost.
  4. Adds a 0/1 flag per hand: "was this hand found in this frame?"
"""
import numpy as np

SEQ_LEN = 60
RAW_DIM = 126
FEATURE_DIM = 132  # 126 relative + 4 wrist x,y + 2 present flags


def tidy_hands(X):
    X = np.asarray(X, dtype=np.float32)
    single = X.ndim == 2
    if single:
        X = X[None]
    N, T, D = X.shape
    if D != RAW_DIM:
        raise ValueError(f"expected {RAW_DIM} numbers per frame, got {D}")

    h = X.reshape(N, T, 2, 21, 3)
    present = np.abs(h).sum(axis=(3, 4)) > 0                      # (N,T,2)
    wrist_x = h[:, :, :, 0, 0]

    # swap blocks when: only block 1 has a hand, or both do and block 1 is further left
    swap = (present[:, :, 1] & ~present[:, :, 0]) | (
        present[:, :, 0] & present[:, :, 1] & (wrist_x[:, :, 1] < wrist_x[:, :, 0])
    )
    h = np.where(swap[:, :, None, None, None], h[:, :, ::-1], h)
    present = np.where(swap[:, :, None], present[:, :, ::-1], present)

    wrist = h[:, :, :, 0, :]                                       # (N,T,2,3)
    rel = h - wrist[:, :, :, None, :]
    size = np.abs(rel).max(axis=(3, 4), keepdims=True) + 1e-6
    rel = (rel / size) * present[:, :, :, None, None]
    wrist_xy = wrist[:, :, :, :2] * present[:, :, :, None]

    feats = np.concatenate(
        [rel.reshape(N, T, 126), wrist_xy.reshape(N, T, 4), present.astype(np.float32)],
        axis=2,
    ).astype(np.float32)
    return feats[0] if single else feats
