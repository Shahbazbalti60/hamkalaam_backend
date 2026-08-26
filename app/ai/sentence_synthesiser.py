"""
app/ai/sentence_synthesiser.py

DUMMY / RULE-BASED PLACEHOLDER — Phase 4.

Real pipeline (Phase 6) trains an LSTM with beam search over word
sequences to produce grammatically natural Urdu sentences. This
placeholder just joins words with a couple of simple pattern rules
so the /generate endpoint has a realistic-shaped response to test
against while the real model doesn't exist yet.
"""


class SentenceSynthesiser:
    """Dummy LSTM stand-in. Replace with a real trained
    sequence-to-sequence model in Phase 6."""

    def generate(self, word_sequence, expression_context=None):
        """
        Args:
            word_sequence: list[str] — Urdu words in signed order.
            expression_context: str | None — e.g. 'Question', 'Happy'.

        Returns:
            urdu_sentence: str
        """
        if not word_sequence:
            return ""

        joined = " ".join(word_sequence)

        if expression_context == "Question":
            return f"کیا {joined}؟"
        if expression_context == "Negation":
            return f"{joined} نہیں۔"

        return f"{joined}۔"