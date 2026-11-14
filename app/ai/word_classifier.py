# -*- coding: utf-8 -*-
"""
app/ai/word_classifier.py

The word recogniser of HamKalaam is the Random Forest of the SRDS
(app/ai/forest_classifier.py). This small factory is the one place that
creates it, so the rest of the app does not care how it is built.
"""
from app.ai.forest_classifier import ForestSequenceClassifier


def build_word_classifier():
    return ForestSequenceClassifier()
