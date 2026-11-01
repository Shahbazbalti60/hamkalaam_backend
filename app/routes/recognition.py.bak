"""
app/routes/recognition.py

Recognition pipeline endpoints for HamKalaam.

IMPORTANT: This uses DUMMY/RULE-BASED placeholder classifiers (see
app/ai/*.py) until real trained models exist (Phase 6). The point of
this phase is to prove the pipeline SHAPE works end-to-end — request
in, prediction out, in the right JSON format — not to produce
accurate PSL recognition yet.

Endpoints:
    POST /api/recognition/classify
        Input:  {hand_landmarks: [float], face_landmarks: [float]}
        Output: {word: str, confidence: float, expression: str}

    POST /api/recognition/generate
        Input:  {word_sequence: [str], expression_context: str}
        Output: {urdu_sentence: str}
"""

from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required
from marshmallow import Schema, fields, ValidationError, validate

from app.ai.gesture_classifier import GestureClassifier
from app.ai.expression_classifier import ExpressionClassifier
from app.ai.sentence_synthesiser import SentenceSynthesiser
from app.ai.sequence_classifier import SequenceClassifier, ModelUnavailable

recognition_bp = Blueprint("recognition", __name__, url_prefix="/api/recognition")

# Instantiated once at import time — these are cheap dummy objects for
# now. When real models are loaded in Phase 6, this is also where
# you'd load the .pkl/.h5 files once at startup rather than per request.
_gesture_classifier = GestureClassifier()
_expression_classifier = ExpressionClassifier()
_sentence_synthesiser = SentenceSynthesiser()
_sequence_classifier = SequenceClassifier()   # real trained model, loaded on first use


# ─────────────────────────────────────────────────────────────
# Validation schemas
# ─────────────────────────────────────────────────────────────

class ClassifySchema(Schema):
    # Real pipeline expects exactly 126 hand values and 136 face
    # values, but we validate loosely here (just "a list of numbers")
    # since the dummy classifiers don't care about exact length, and
    # over-strict validation would make early pipeline testing
    # brittle. Tighten this once real models with fixed input shapes
    # are wired in (Phase 6).
    hand_landmarks = fields.List(fields.Float(), required=True)
    face_landmarks = fields.List(fields.Float(), required=False, load_default=[])


class GenerateSchema(Schema):
    word_sequence = fields.List(
        fields.Str(), required=True, validate=validate.Length(min=1)
    )
    expression_context = fields.Str(required=False, allow_none=True)


classify_schema = ClassifySchema()
generate_schema = GenerateSchema()


# ─────────────────────────────────────────────────────────────
# POST /api/recognition/classify
# ─────────────────────────────────────────────────────────────

@recognition_bp.route("/classify", methods=["POST"])
@jwt_required()
def classify():
    json_data = request.get_json(silent=True) or {}

    try:
        data = classify_schema.load(json_data)
    except ValidationError as e:
        return jsonify({"error": "Validation failed", "details": e.messages}), 400

    word, confidence = _gesture_classifier.predict(data["hand_landmarks"])
    expression = _expression_classifier.predict(data["face_landmarks"])

    return jsonify({
        "word": word,
        "confidence": confidence,
        "expression": expression,
    }), 200


# ─────────────────────────────────────────────────────────────
# POST /api/recognition/generate
# ─────────────────────────────────────────────────────────────

@recognition_bp.route("/generate", methods=["POST"])
@jwt_required()
def generate():
    json_data = request.get_json(silent=True) or {}

    try:
        data = generate_schema.load(json_data)
    except ValidationError as e:
        return jsonify({"error": "Validation failed", "details": e.messages}), 400

    sentence = _sentence_synthesiser.generate(
        data["word_sequence"],
        data.get("expression_context"),
    )

    return jsonify({"urdu_sentence": sentence}), 200


# ─────────────────────────────────────────────────────────────
# REAL model: one signing window (about 60 frames) -> a word
# ─────────────────────────────────────────────────────────────

class SequenceSchema(Schema):
    # ~60 frames, each frame = 126 numbers (2 hands x 21 points x x,y,z).
    # Any length from 5 to 300 frames is accepted; the server resamples to 60.
    frames = fields.List(
        fields.List(fields.Float(), validate=validate.Length(equal=126)),
        required=True,
        validate=validate.Length(min=5, max=300),
    )


sequence_schema = SequenceSchema()


@recognition_bp.route("/model-status", methods=["GET"])
@jwt_required()
def model_status():
    """Loads the model (first call takes a few seconds) so the app can
    preload it when the recognition screen opens."""
    try:
        _sequence_classifier.load()
    except ModelUnavailable as exc:
        return jsonify({"ready": False, "reason": str(exc)}), 200
    return jsonify({"ready": True, "word_count": len(_sequence_classifier.words)}), 200


@recognition_bp.route("/classify-sequence", methods=["POST"])
@jwt_required()
def classify_sequence():
    json_data = request.get_json(silent=True) or {}
    try:
        data = sequence_schema.load(json_data)
    except ValidationError as e:
        return jsonify({"error": "Validation failed", "details": e.messages}), 400

    try:
        result = _sequence_classifier.predict(data["frames"])
    except ModelUnavailable as exc:
        return jsonify({"error": str(exc)}), 503
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    return jsonify(result), 200
