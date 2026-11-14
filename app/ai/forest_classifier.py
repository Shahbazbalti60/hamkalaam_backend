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


class ForestSequenceClassifier:
    KIND = "forest"
    MODEL_FILE = MODEL_FILE
    LABELS_FILE = LABELS_FILE
    DESCRIPTION = "Random Forest on clip features (SRDS word classifier)"
    SEQ_LEN = 60
    MIN_HANDS_RATIO = 0.03
    CONFIDENT = 0.30          # replaced by the value measured in training (forest_labels.json)

    def __init__(self, model_dir=None):
        self._model_dir = model_dir or MODEL_DIR
        self._model = None
        self._words = None
        self._lock = threading.Lock()

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
        row = clip_features(tidy_hands(arr))[None]
        with self._lock:
            probs = self._model.predict_proba(row)[0]
            classes = np.asarray(self._model.classes_, dtype=int)

        order = np.argsort(probs)[::-1][:3]
        top3 = [{"word": self._words[int(classes[i])], "confidence": round(float(probs[i]), 4)}
                for i in order]
        best = top3[0]
        return {"word": best["word"], "confidence": best["confidence"], "top3": top3,
                "hands_found": round(hands_found, 3),
                "uncertain": best["confidence"] < self.CONFIDENT, "reason": None}
