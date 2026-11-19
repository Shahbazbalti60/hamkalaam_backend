# -*- coding: utf-8 -*-
"""
app/ai/forest_classifier.py   (SRDS: Random Forest word classifier)

The word recogniser of the app. Its answer has this shape:

    predict(frames) -> {"word", "confidence", "top3", "hands_found", "uncertain", "reason"}

Files (made by the forest training notebook, copied into app/ai/model/):
    psl_forest.joblib      the trained forest
    forest_labels.json     {"words": [...], "confident": 0.xx}

scikit-learn is imported only when the model is first needed.
"""
import json
import os
import threading
from collections import deque
from datetime import datetime

import numpy as np

try:
    from .hand_tidy import tidy_hands
    from .forest_features import clip_features
except ImportError:                      # plain-script use / tests
    from hand_tidy import tidy_hands
    from forest_features import clip_features

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
MODEL_FILE = "psl_forest.joblib"
LABELS_FILE = "forest_labels.json"
RESULTS_FILE = "forest_results.json"


class ModelUnavailable(Exception):
    """scikit-learn or the forest files are missing."""


class OrientationTracker:
    """Decides, clip by clip, whether the live picture is mirrored compared with the training data.

    Both versions of a clip (as recorded / mirrored) are scored. Until the last few confident clips
    agree, the version the forest is more sure about is used. Once at least 75 % of the last 20
    confident clips preferred the same version, that version is used for every clip (so one odd
    clip cannot flip the answer). The memory is short, so switching between a laptop and a phone
    camera corrects itself after a few signs.
    """
    WINDOW = 20
    MIN_VOTES = 6
    LOCK_SHARE = 0.75
    CONFIDENT = 0.5

    def __init__(self):
        self._recent = deque(maxlen=self.WINDOW)    # True = the mirrored version won
        self._lock = threading.Lock()

    def decide(self, plain_best, flipped_best):
        """True = use the mirrored version of this clip."""
        with self._lock:
            if len(self._recent) >= self.MIN_VOTES:
                share = sum(self._recent) / len(self._recent)
                if share >= self.LOCK_SHARE:
                    return True
                if share <= 1 - self.LOCK_SHARE:
                    return False
            return flipped_best > plain_best

    def learn(self, plain_best, flipped_best):
        if max(plain_best, flipped_best) >= self.CONFIDENT:
            with self._lock:
                self._recent.append(flipped_best > plain_best)


def mirror_clip(arr):
    """The same clip as seen in a mirror: x -> 1 - x for every landmark that exists."""
    h = np.asarray(arr, dtype=np.float32).reshape(-1, 2, 21, 3).copy()
    present = np.abs(h).sum(axis=-1) > 0
    h[..., 0] = np.where(present, 1.0 - h[..., 0], 0.0)
    return h.reshape(-1, 126)


class ForestSequenceClassifier:
    KIND = "forest"
    MODEL_FILE = MODEL_FILE
    LABELS_FILE = LABELS_FILE
    DESCRIPTION = "Random Forest on clip features (SRDS word classifier)"
    SEQ_LEN = 60
    MIN_HANDS_RATIO = 0.03
    CONFIDENT = 0.30          # replaced by the value measured in training (forest_labels.json)
    # The training clips and the live camera may see the signer mirrored relative to each other
    # (the forest drops to ~30 % on a mirrored picture). MIRROR_INPUT in .env:
    #   auto (default) - found out from the clips themselves (see OrientationTracker)
    #   yes / no       - always / never use the mirrored picture

    def __init__(self, model_dir=None):
        self._model_dir = model_dir or MODEL_DIR
        self._model = None
        self._words = None
        self._lock = threading.Lock()
        self._orientation = OrientationTracker()

    @property
    def is_loaded(self):
        return self._model is not None

    @property
    def words(self):
        return list(self._words or [])

    def load(self):
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            model_path = os.path.join(self._model_dir, MODEL_FILE)
            labels_path = os.path.join(self._model_dir, LABELS_FILE)
            if not (os.path.exists(model_path) and os.path.exists(labels_path)):
                raise ModelUnavailable(
                    f"Random Forest files not found in {self._model_dir} "
                    f"(need {MODEL_FILE} and {LABELS_FILE}).")
            try:
                import joblib
            except ImportError as exc:
                raise ModelUnavailable(
                    "scikit-learn is not installed (run: pip install scikit-learn).") from exc
            try:
                model = joblib.load(model_path)
            except Exception as exc:
                raise ModelUnavailable(
                    f"The forest file could not be loaded ({exc}). Install the same "
                    "scikit-learn version that was used for training.") from exc
            with open(labels_path, "r", encoding="utf-8") as f:
                info = json.load(f)
            self._words = info["words"]
            if isinstance(info.get("confident"), (int, float)):
                self.CONFIDENT = float(info["confident"])
            self._model = model

    def predict(self, frames):
        arr = np.asarray(frames, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != 126 or arr.shape[0] < 5:
            raise ValueError("frames must be a list of at least 5 frames of 126 numbers")
        if not np.isfinite(arr).all():
            raise ValueError("frames contain invalid numbers")

        if arr.shape[0] != self.SEQ_LEN:
            idx = np.round(np.linspace(0, arr.shape[0] - 1, self.SEQ_LEN)).astype(int)
            arr = arr[idx]

        hands_found = float((np.abs(arr).sum(axis=1) > 0).mean())
        if hands_found < self.MIN_HANDS_RATIO:
            return {"word": None, "confidence": 0.0, "top3": [],
                    "hands_found": hands_found, "uncertain": True, "reason": "no_hand"}

        self.load()
        rows = np.stack([clip_features(tidy_hands(arr)),
                         clip_features(tidy_hands(mirror_clip(arr)))])
        with self._lock:
            both = self._model.predict_proba(rows)
            classes = np.asarray(self._model.classes_, dtype=int)
        plain_best, flipped_best = float(both[0].max()), float(both[1].max())
        mode = (os.environ.get("MIRROR_INPUT") or "auto").strip().lower()
        if mode in ("yes", "true", "1"):
            mirrored = True
        elif mode in ("no", "false", "0"):
            mirrored = False
        else:
            mirrored = self._orientation.decide(plain_best, flipped_best)
            self._orientation.learn(plain_best, flipped_best)
        probs = both[1] if mirrored else both[0]

        order = np.argsort(probs)[::-1][:3]
        top3 = [{"word": self._words[int(classes[i])], "confidence": round(float(probs[i]), 4)}
                for i in order]
        best = top3[0]
        result = {"word": best["word"], "confidence": best["confidence"], "top3": top3,
                  "hands_found": round(hands_found, 3),
                  "uncertain": best["confidence"] < self.CONFIDENT, "reason": None,
                  "mirrored": mirrored}
        self._log_live(arr, result)
        return result

    def _log_live(self, arr, result):
        """Only when LOG_LIVE_SAMPLES=1 in .env: keeps every live clip (and what the
        model said) in uploads/live_samples/samples.jsonl so live data can be compared
        with the training data. Never raises."""
        if (os.environ.get("LOG_LIVE_SAMPLES") or "").strip().lower() not in ("1", "true", "yes"):
            return
        try:
            root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
            folder = os.path.join(root, "uploads", "live_samples")
            os.makedirs(folder, exist_ok=True)
            row = {"t": datetime.utcnow().isoformat() + "Z", "word": result["word"],
                   "confidence": result["confidence"], "top3": result["top3"],
                   "frames": np.round(arr, 4).tolist()}
            with open(os.path.join(folder, "samples.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
        except Exception:
            pass
