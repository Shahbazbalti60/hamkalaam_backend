# -*- coding: utf-8 -*-
"""
app/ai/forest_features.py

Turns ONE signed word (60 frames x 132 numbers, the output of hand_tidy.tidy_hands)
into ONE row of numbers for the Random Forest (SRDS: Random Forest word classifier).

A Random Forest looks at one table row at a time, so the clip is summarised:
  mean, std, min, max of every number over the 60 frames  (where the hand is, how big the movement is)
  mean of the first, middle and last third of the clip     (what the hand does over time)
  average frame-to-frame change                            (how much and how fast it moves)
8 summaries x 132 numbers = 1056 features.

The SAME file is used in the training notebook and on the server.
"""
import numpy as np

N_FRAMES = 60
N_TIDY = 132
N_FEATURES = 8 * N_TIDY


def clip_features(tidy):
    """tidy: (60, 132) array -> (1056,) float32 feature row."""
    t = np.asarray(tidy, dtype=np.float32)
    if t.ndim != 2 or t.shape[1] != N_TIDY:
        raise ValueError(f"expected a (frames, {N_TIDY}) array, got {t.shape}")
    thirds = np.array_split(t, 3, axis=0)
    parts = [t.mean(0), t.std(0), t.min(0), t.max(0)]
    parts += [x.mean(0) for x in thirds]
    parts.append(np.abs(np.diff(t, axis=0)).mean(0))
    return np.concatenate(parts).astype(np.float32)
