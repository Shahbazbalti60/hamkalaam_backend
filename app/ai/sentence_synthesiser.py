# -*- coding: utf-8 -*-
"""
app/ai/sentence_synthesiser.py

Turns the recognised PSL word labels (e.g. ["water", "want"]) plus the
facial-expression context into an Urdu sentence.

Day 4: now backed by the rule-based engine in sentence_engine.py
(tested by test_sentence_engine.py). The class name and the generate()
signature are unchanged, so routes/recognition.py needs no edit.
A learned LSTM sequence-to-sequence model can replace the engine later
without touching the route.
"""
from app.ai import sentence_engine


class SentenceSynthesiser:
    """Word labels + expression context -> Urdu sentence."""

    def generate(self, word_sequence, expression_context=None):
        """
        Args:
            word_sequence: list[str] - PSL word labels in signed order
                (the labels the classifier returns, e.g. "water").
            expression_context: str | None - 'Question', 'Negation', ...

        Returns:
            urdu_sentence: str ("" if word_sequence is empty)
        """
        return sentence_engine.generate(word_sequence, expression_context)
