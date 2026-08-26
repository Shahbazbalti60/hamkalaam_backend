"""
app/ai/expression_classifier.py

DUMMY / RULE-BASED PLACEHOLDER — Phase 4.

Same caveat as gesture_classifier.py: this is NOT a trained CNN. It
exists to let the /classify endpoint return a plausible-shaped
response (word + expression) so the frontend and pipeline can be
built and tested before a real facial-expression model is trained
in Phase 6.
"""

EXPRESSION_BANK = [
    "Neutral",
    "Question",
    "Happy",
    "Sad",
    "Negation",
    "Affirmation",
]


class ExpressionClassifier:
    """Dummy CNN stand-in. Replace with a real trained model in
    Phase 6."""

    def predict(self, face_landmarks):
        """
        Args:
            face_landmarks: list of floats. Real pipeline expects 136
                values, but this dummy version accepts any list
                (including empty, which returns 'Neutral').

        Returns:
            expression: str — one of EXPRESSION_BANK.
        """
        if not face_landmarks:
            return "Neutral"

        index = int(abs(sum(face_landmarks)) * 1000) % len(EXPRESSION_BANK)
        return EXPRESSION_BANK[index]