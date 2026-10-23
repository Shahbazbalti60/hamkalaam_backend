"""
app/ai/sequence_classifier.py

Real word recogniser (trained in Google Colab on the SignSpeak PSL
landmark dataset, Apache-2.0, Sherwani et al.).

Input : one signing window = ~60 frames x 126 numbers
        (2 hand blocks x 21 points x x,y,z, exactly like the dataset)
Output: best word + confidence + top-3 list

The same hand_tidy.py used for training prepares the numbers, so live
camera data and training data are treated identically.

TensorFlow is imported only when the model is first needed, so Flask
still starts quickly (and still starts at all) if TensorFlow is missing.
"""
import json
import os
import threading

import numpy as np

try:
    from .hand_tidy import tidy_hands
except ImportError:                      # plain-script use / tests
    from hand_tidy import tidy_hands

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
MODEL_FILE = "psl_lstm.keras"
LABELS_FILE = "labels.json"


class ModelUnavailable(Exception):
    """TensorFlow or the model files are missing."""


class SequenceClassifier:
    SEQ_LEN = 60
    MIN_HANDS_RATIO = 0.15      # fewer frames than this with a hand -> "no hand"
    CONFIDENT = 0.50            # below this the answer is flagged "uncertain"

    def __init__(self, model_dir=None):
        self._model_dir = model_dir or MODEL_DIR
        self._model = None
        self._words = None
        self._lock = threading.Lock()

    # ---------------------------------------------------------- loading
    @property
    def is_loaded(self):
        return self._model is not None

    def load(self):
        """Load the model once (safe to call many times)."""
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            model_path = os.path.join(self._model_dir, MODEL_FILE)
            labels_path = os.path.join(self._model_dir, LABELS_FILE)
            if not (os.path.exists(model_path) and os.path.exists(labels_path)):
                raise ModelUnavailable(
                    f"Model files not found in {self._model_dir} "
                    f"(need {MODEL_FILE} and {LABELS_FILE})."
                )
            try:
                from tensorflow import keras
            except ImportError as exc:
                raise ModelUnavailable(
                    "TensorFlow is not installed in this environment "
                    "(run: pip install tensorflow)."
                ) from exc
            model = keras.models.load_model(model_path)
            with open(labels_path, "r", encoding="utf-8") as f:
                words = json.load(f)["words"]
            self._words = words
            self._model = model

    @property
    def words(self):
        return list(self._words or [])

    # ---------------------------------------------------------- predict
    def predict(self, frames):
        arr = np.asarray(frames, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != 126 or arr.shape[0] < 5:
            raise ValueError("frames must be a list of at least 5 frames of 126 numbers")
        if not np.isfinite(arr).all():
            raise ValueError("frames contain invalid numbers")

        # Any length is accepted: stretch / squeeze to exactly 60 frames.
        if arr.shape[0] != self.SEQ_LEN:
            idx = np.round(np.linspace(0, arr.shape[0] - 1, self.SEQ_LEN)).astype(int)
            arr = arr[idx]

        hands_found = float((np.abs(arr).sum(axis=1) > 0).mean())
        if hands_found < self.MIN_HANDS_RATIO:
            return {"word": None, "confidence": 0.0, "top3": [],
                    "hands_found": hands_found, "uncertain": True, "reason": "no_hand"}

        self.load()
        feats = tidy_hands(arr)[None]
        with self._lock:                                  # one prediction at a time
            probs = self._model.predict(feats, verbose=0)[0]

        order = np.argsort(probs)[::-1][:3]
        top3 = [{"word": self._words[i], "confidence": round(float(probs[i]), 4)} for i in order]
        best = top3[0]
        return {"word": best["word"], "confidence": best["confidence"], "top3": top3,
                "hands_found": round(hands_found, 3),
                "uncertain": best["confidence"] < self.CONFIDENT, "reason": None}
