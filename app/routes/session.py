"""
app/routes/session.py

Session history endpoints for HamKalaam.
A "Session" represents one continuous recognition session (camera on ->
camera off). Each session can contain many SessionSentence rows — the
Urdu sentences generated during that session.

Ownership rule: a normal user can only see/modify their OWN sessions.
Admins can see/delete any session (useful for support/debugging).

Endpoints:
    POST   /api/session/start     -> create new session for current user
    POST   /api/session/sentence  -> save a generated sentence to a session
    GET    /api/session/history   -> get current user's sessions (+ sentences)
    DELETE /api/session/<id>      -> delete a session (+ its sentences)
"""

from datetime import datetime

from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from marshmallow import Schema, fields, ValidationError, validate

from app import db
from app.models.user import User, Session, SessionSentence

session_bp = Blueprint("session", __name__, url_prefix="/api/session")


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _current_user():
    """Return the User row for the JWT identity, or None."""
    user_id = get_jwt_identity()
    return db.session.get(User, user_id)


def _is_admin(user):
    # Role is stored as 'User' / 'Admin' (capitalized) in this project,
    # so compare case-insensitively rather than assuming exact casing.
    return (user.role or "").lower() == "admin"


def _serialize_sentence(s):
    return {
        "id": s.id,
        "session_id": s.session_id,
        "urdu_sentence": s.urdu_sentence,
        "word_sequence": s.word_sequence,
        "expression_context": s.expression_context,
        "generated_at": s.generated_at.isoformat() if s.generated_at else None,
    }


def _serialize_session(sess, include_sentences=True):
    data = {
        "id": sess.id,
        "user_id": sess.user_id,
        "started_at": sess.started_at.isoformat() if sess.started_at else None,
        "ended_at": sess.ended_at.isoformat() if sess.ended_at else None,
    }
    if include_sentences:
        data["sentences"] = [_serialize_sentence(s) for s in sess.sentences]
        data["sentence_count"] = len(sess.sentences)
    return data


# ─────────────────────────────────────────────────────────────
# Validation schemas
# ─────────────────────────────────────────────────────────────

class SentenceSchema(Schema):
    session_id = fields.Int(required=True)
    urdu_sentence = fields.Str(required=True, validate=validate.Length(min=1))
    # word_sequence is stored as a plain String(500) in the model (not JSON),
    # so we accept a string here. If the frontend sends a list, join it.
    word_sequence = fields.Str(required=False, allow_none=True, validate=validate.Length(max=500))
    expression_context = fields.Str(required=False, allow_none=True, validate=validate.Length(max=50))


sentence_schema = SentenceSchema()


# ─────────────────────────────────────────────────────────────
# POST /api/session/start
# ─────────────────────────────────────────────────────────────

@session_bp.route("/start", methods=["POST"])
@jwt_required()
def start_session():
    user = _current_user()
    if user is None:
        return jsonify({"error": "User not found"}), 404

    new_session = Session(
        user_id=user.id,
        started_at=datetime.utcnow(),
        ended_at=None,
    )
    db.session.add(new_session)
    db.session.commit()

    return jsonify({
        "message": "Session started",
        "session": _serialize_session(new_session, include_sentences=False),
    }), 201


# ─────────────────────────────────────────────────────────────
# POST /api/session/sentence
# ─────────────────────────────────────────────────────────────

@session_bp.route("/sentence", methods=["POST"])
@jwt_required()
def save_sentence():
    user = _current_user()
    if user is None:
        return jsonify({"error": "User not found"}), 404

    json_data = request.get_json(silent=True) or {}

    # Allow word_sequence to arrive as a list (e.g. ["ghar","jana"]) and
    # normalize it to a comma-separated string for storage, since the
    # model column is a plain String(500), not JSON.
    if isinstance(json_data.get("word_sequence"), list):
        json_data["word_sequence"] = ",".join(str(w) for w in json_data["word_sequence"])

    try:
        data = sentence_schema.load(json_data)
    except ValidationError as e:
        return jsonify({"error": "Validation failed", "details": e.messages}), 400

    target_session = db.session.get(Session, data["session_id"])
    if target_session is None:
        return jsonify({"error": "Session not found"}), 404

    # A user can only add sentences to their OWN session.
    if target_session.user_id != user.id and not _is_admin(user):
        return jsonify({"error": "You do not have access to this session"}), 403

    sentence = SessionSentence(
        session_id=target_session.id,
        urdu_sentence=data["urdu_sentence"],
        word_sequence=data.get("word_sequence"),
        expression_context=data.get("expression_context"),
        generated_at=datetime.utcnow(),
    )
    db.session.add(sentence)
    db.session.commit()

    return jsonify({
        "message": "Sentence saved",
        "sentence": _serialize_sentence(sentence),
    }), 201


# ─────────────────────────────────────────────────────────────
# GET /api/session/history
# ─────────────────────────────────────────────────────────────

@session_bp.route("/history", methods=["GET"])
@jwt_required()
def get_history():
    user = _current_user()
    if user is None:
        return jsonify({"error": "User not found"}), 404

    sessions = (
        Session.query
        .filter_by(user_id=user.id)
        .order_by(Session.started_at.desc())
        .all()
    )

    return jsonify({
        "sessions": [_serialize_session(s) for s in sessions],
        "total": len(sessions),
    }), 200


# ─────────────────────────────────────────────────────────────
# DELETE /api/session/<id>
# ─────────────────────────────────────────────────────────────

@session_bp.route("/<int:session_id>", methods=["DELETE"])
@jwt_required()
def delete_session(session_id):
    user = _current_user()
    if user is None:
        return jsonify({"error": "User not found"}), 404

    target_session = db.session.get(Session, session_id)
    if target_session is None:
        return jsonify({"error": "Session not found"}), 404

    # A user can only delete their OWN session; admins can delete any.
    if target_session.user_id != user.id and not _is_admin(user):
        return jsonify({"error": "You do not have access to this session"}), 403

    # Remove dependent sentences first (no cascade assumed on the FK).
    SessionSentence.query.filter_by(session_id=target_session.id).delete()
    db.session.delete(target_session)
    db.session.commit()

    return jsonify({"message": "Session deleted"}), 200