"""
app/ai/gesture_classifier.py

DUMMY / RULE-BASED PLACEHOLDER — Phase 4.

This is NOT a trained model. It exists purely to let the rest of the
pipeline (Flask endpoint -> Flutter UI) be built and tested end-to-end
before real PSL video data is collected and a real Random Forest
classifier is trained (Phase 6).

The "prediction" here is deterministic (same landmarks always produce
the same word) so behavior is testable and reproducible, but it has
NO relationship to actual PSL gestures. Do not present this as
working gesture recognition — it is a pipeline-shape placeholder only.
"""

WORD_BANK = [
    "پانی",   # water
    "دینا",   # give
    "مجھے",   # to me
    "شکریہ",  # thank you
    "مدد",    # help
    "گھر",    # home
    "کھانا",  # food
    "ڈاکٹر",  # doctor
]


class GestureClassifier:
    """Dummy Random-Forest stand-in. Replace with a real trained
    model in Phase 6 (see scripts/extract_landmarks.py + Colab
    training notes in the FYP plan)."""

    def predict(self, hand_landmarks):
        """
        Args:
            hand_landmarks: list of floats. Real pipeline expects 126
                values (2 hands x 21 landmarks x 3 coords), but this
                dummy version accepts any non-empty list.

        Returns:
            (word: str, confidence: float) — confidence is a fixed
            placeholder, not a real model probability.
        """
        if not hand_landmarks:
            return None, 0.0

        # Deterministic pseudo-selection so the same input always
        # returns the same word — useful for automated testing.
        index = int(abs(sum(hand_landmarks)) * 1000) % len(WORD_BANK)
        return WORD_BANK[index], 0.75