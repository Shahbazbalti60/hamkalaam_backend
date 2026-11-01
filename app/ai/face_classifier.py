# -*- coding: utf-8 -*-
"""
app/ai/face_classifier.py   (SRDS FR-21)

Facial-expression CNN: 68 face landmarks -> one of six classes
(Question, Negation, Affirmation, Neutral, Happy, Sad).

Files (trained in the Colab notebook, copied into app/ai/model/):
    psl_face_cnn.keras      the trained network
    face_labels.json        {"classes": [...]} in the network's output order

TensorFlow is loaded only when the model is first needed, exactly like
sequence_classifier.py, so the server still starts without it.
"""
import json
import os
import threading

import numpy as np

from app.ai.face_landmarks import VECTOR_LEN, vector_to_input

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
MODEL_FILE = "psl_face_cnn.keras"
LABELS_FILE = "face_labels.json"
MAX_BATCH = 40

CLASSES = ["Question", "Negation", "Affirmation", "Neutral", "Happy", "Sad"]


class FaceModelUnavailable(Exception):
    """TensorFlow or the face model files are missing."""


class FaceClassifier:
    def __init__(self, model_dir=None):
        self._model_dir = model_dir or MODEL_DIR
        self._model = None
        self._classes = None
        self._lock = threading.Lock()

    @property
    def is_loaded(self):
        return self._model is not None

    @property
    def classes(self):
        return list(self._classes or CLASSES)

    def files_present(self):
        return (os.path.exists(os.path.join(self._model_dir, MODEL_FILE))
                and os.path.exists(os.path.join(self._model_dir, LABELS_FILE)))

    def load(self):
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            model_path = os.path.join(self._model_dir, MODEL_FILE)
            labels_path = os.path.join(self._model_dir, LABELS_FILE)
            if not (os.path.exists(model_path) and os.path.exists(labels_path)):
                raise FaceModelUnavailable(
                    "The face-expression model has not been trained yet "
                    f"(need {MODEL_FILE} and {LABELS_FILE} in app/ai/model/).")
            try:
                from tensorflow import keras
            except ImportError as exc:
                raise FaceModelUnavailable(
                    "TensorFlow is not installed in this environment.") from exc
            model = keras.models.load_model(model_path)
            with open(labels_path, "r", encoding="utf-8") as f:
                classes = json.load(f)["classes"]
            self._classes = classes
            self._model = model

    def predict(self, vectors):
        """vectors: list of 136-number lists (raw pixel face points).
        Returns [{"label": str, "confidence": float}, ...] in the same order."""
        if not isinstance(vectors, (list, tuple)) or not vectors:
            raise ValueError("send at least one face vector")
        if len(vectors) > MAX_BATCH:
            raise ValueError(f"send at most {MAX_BATCH} face vectors at a time")

        batch = []
        for v in vectors:
            if not isinstance(v, (list, tuple)) or len(v) != VECTOR_LEN:
                raise ValueError(f"each face vector must have {VECTOR_LEN} numbers")
            batch.append(vector_to_input(v))
        x = np.stack(batch).astype(np.float32)

        self.load()
        with self._lock:
            probs = self._model.predict(x, verbose=0)

        out = []
        for row in probs:
            i = int(np.argmax(row))
            out.append({"label": self._classes[i], "confidence": round(float(row[i]), 4)})
        return out
