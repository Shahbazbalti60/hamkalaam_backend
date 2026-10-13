"""
app/routes/dataset.py

Dataset management endpoints for HamKalaam.
Handles the taxonomy of PSL words (DatasetWord) and the samples
collected for each word (DatasetSample). All endpoints require a
valid JWT; mutating endpoints (POST/DELETE) are admin-only.

Endpoints:
    POST   /api/dataset/word         -> add new PSL word
    GET    /api/dataset/words        -> all words + sample counts
    DELETE /api/dataset/word/<id>    -> delete a word (+ its samples)
    GET    /api/dataset/statistics   -> stats for the admin dashboard
    POST   /api/dataset/word/<id>/sample -> upload one recorded sample video
"""

import os
import shutil
import uuid

from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from marshmallow import Schema, fields, ValidationError, validate
from sqlalchemy.exc import IntegrityError

from app import db
from app.models.user import User, DatasetWord, DatasetSample

# NOTE: url_prefix is set here AND when registering in app/__init__.py's
# auth_bp registration style used '/api/auth' at register time. Pick ONE
# place to set it, not both, or routes will double up (e.g. /api/dataset/api/dataset/word).
# This file sets it here; register with app.register_blueprint(dataset_bp) — no url_prefix arg.
dataset_bp = Blueprint("dataset", __name__, url_prefix="/api/dataset")

# Where recorded sample videos are stored on the server's disk:
#   <backend root>/uploads/dataset_samples/<word_id>/<random>.webm|mp4
# (add "uploads/" to .gitignore so videos never end up in git)
_BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
UPLOAD_ROOT = os.path.join(_BACKEND_ROOT, "uploads", "dataset_samples")
ALLOWED_VIDEO_EXTENSIONS = {"webm", "mp4", "mov", "mkv"}


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _current_user():
    """Return the User row for the JWT identity, or None."""
    user_id = get_jwt_identity()
    return db.session.get(User, user_id)


def _require_admin():
    """
    Returns None if the current user is an admin.
    Otherwise returns a (response, status_code) tuple to short-circuit
    the route, e.g.:  if err := _require_admin(): return err
    """
    user = _current_user()
    if user is None:
        return jsonify({"error": "User not found"}), 404
    # User.role defaults to 'User' (capitalized) in the model, so compare
    # case-insensitively rather than assuming a lowercase 'admin' string.
    if (user.role or "").lower() != "admin":
        return jsonify({"error": "Admin access required"}), 403
    return None


# ─────────────────────────────────────────────────────────────
# Validation schema
# ─────────────────────────────────────────────────────────────

class DatasetWordSchema(Schema):
    # Model column is String(100) — keep max lengths in sync with user.py.
    urdu_word = fields.Str(required=True, validate=validate.Length(min=1, max=100))
    # english_gloss/domain are nullable in the model, so not required here,
    # but still validated if provided.
    english_gloss = fields.Str(required=False, allow_none=True, validate=validate.Length(max=100))
    domain = fields.Str(required=False, allow_none=True, validate=validate.Length(max=50))


word_schema = DatasetWordSchema()


# ─────────────────────────────────────────────────────────────
# POST /api/dataset/word
# ─────────────────────────────────────────────────────────────

@dataset_bp.route("/word", methods=["POST"])
@jwt_required()
def add_word():
    if err := _require_admin():
        return err

    json_data = request.get_json(silent=True) or {}

    try:
        data = word_schema.load(json_data)
    except ValidationError as e:
        return jsonify({"error": "Validation failed", "details": e.messages}), 400

    # Prevent duplicate Urdu words in the taxonomy
    existing = DatasetWord.query.filter_by(urdu_word=data["urdu_word"]).first()
    if existing:
        return jsonify({"error": "This word already exists in the dataset"}), 409

    word = DatasetWord(
        urdu_word=data["urdu_word"],
        english_gloss=data.get("english_gloss"),
        domain=data.get("domain"),
        sample_count=0,
    )
    db.session.add(word)

    try:
        db.session.commit()
    except IntegrityError:
        # Safety net in case of a race condition past the earlier
        # existence check (urdu_word has a DB-level unique constraint).
        db.session.rollback()
        return jsonify({"error": "This word already exists in the dataset"}), 409

    return jsonify({
        "message": "Word added successfully",
        "word": {
            "id": word.id,
            "urdu_word": word.urdu_word,
            "english_gloss": word.english_gloss,
            "domain": word.domain,
            "sample_count": word.sample_count,
            "created_at": word.created_at.isoformat() if word.created_at else None,
        }
    }), 201


# ─────────────────────────────────────────────────────────────
# GET /api/dataset/words
# ─────────────────────────────────────────────────────────────

@dataset_bp.route("/words", methods=["GET"])
@jwt_required()
def get_words():
    domain_filter = request.args.get("domain")

    query = DatasetWord.query
    if domain_filter:
        query = query.filter_by(domain=domain_filter)

    words = query.order_by(DatasetWord.created_at.desc()).all()

    result = []
    for w in words:
        # sample_count on the row is a cached counter; recompute live too
        # in case samples were added/removed without updating the counter.
        live_count = DatasetSample.query.filter_by(word_id=w.id).count()
        result.append({
            "id": w.id,
            "urdu_word": w.urdu_word,
            "english_gloss": w.english_gloss,
            "domain": w.domain,
            "sample_count": live_count,
            "created_at": w.created_at.isoformat() if w.created_at else None,
        })

    return jsonify({"words": result, "total": len(result)}), 200


# ─────────────────────────────────────────────────────────────
# DELETE /api/dataset/word/<id>
# ─────────────────────────────────────────────────────────────

@dataset_bp.route("/word/<int:word_id>", methods=["DELETE"])
@jwt_required()
def delete_word(word_id):
    if err := _require_admin():
        return err

    word = db.session.get(DatasetWord, word_id)
    if word is None:
        return jsonify({"error": "Word not found"}), 404

    # Remove dependent samples first (no cascade assumed on the FK).
    DatasetSample.query.filter_by(word_id=word.id).delete()
    db.session.delete(word)
    db.session.commit()

    # Also remove this word's recorded videos from disk.
    shutil.rmtree(os.path.join(UPLOAD_ROOT, str(word_id)), ignore_errors=True)

    return jsonify({"message": f"Word '{word.urdu_word}' and its samples were deleted"}), 200


# ─────────────────────────────────────────────────────────────
# GET /api/dataset/statistics
# ─────────────────────────────────────────────────────────────

@dataset_bp.route("/statistics", methods=["GET"])
@jwt_required()
def get_statistics():
    total_words = DatasetWord.query.count()
    total_samples = DatasetSample.query.count()

    words = DatasetWord.query.all()
    per_word = []
    for w in words:
        count = DatasetSample.query.filter_by(word_id=w.id).count()
        per_word.append({
            "word_id": w.id,
            "urdu_word": w.urdu_word,
            "english_gloss": w.english_gloss,
            "sample_count": count,
        })

    # Words below the 50-sample-per-word target (see FYP dataset plan)
    TARGET_SAMPLES_PER_WORD = 50
    under_target = [p for p in per_word if p["sample_count"] < TARGET_SAMPLES_PER_WORD]

    avg_samples = (total_samples / total_words) if total_words else 0

    return jsonify({
        "total_words": total_words,
        "total_samples": total_samples,
        "average_samples_per_word": round(avg_samples, 2),
        "target_samples_per_word": TARGET_SAMPLES_PER_WORD,
        "words_under_target": len(under_target),
        "per_word": per_word,
    }), 200


# ─────────────────────────────────────────────────────────────
# POST /api/dataset/word/<id>/sample   (multipart/form-data)
# ─────────────────────────────────────────────────────────────

@dataset_bp.route("/word/<int:word_id>/sample", methods=["POST"])
@jwt_required()
def upload_sample(word_id):
    """
    Receives ONE recorded sign video for a word.
    Form fields:
        video            (file, required)  .webm / .mp4 / .mov / .mkv
        duration_seconds (float, optional)
        lighting_level   (float, optional)
    """
    if err := _require_admin():
        return err

    user = _current_user()
    word = db.session.get(DatasetWord, word_id)
    if word is None:
        return jsonify({"error": "Word not found"}), 404

    file = request.files.get("video")
    if file is None or not file.filename:
        return jsonify({"error": "No video file was sent"}), 400

    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in ALLOWED_VIDEO_EXTENSIONS:
        return jsonify({
            "error": "Unsupported video type",
            "allowed": sorted(ALLOWED_VIDEO_EXTENSIONS),
        }), 400

    word_dir = os.path.join(UPLOAD_ROOT, str(word.id))
    os.makedirs(word_dir, exist_ok=True)

    stored_name = f"{uuid.uuid4().hex}.{ext}"
    file.save(os.path.join(word_dir, stored_name))

    sample = DatasetSample(
        word_id=word.id,
        # stored relative to the backend root so it is portable
        file_path=f"uploads/dataset_samples/{word.id}/{stored_name}",
        duration_seconds=request.form.get("duration_seconds", type=float),
        signer_id=user.id if user else None,
        lighting_level=request.form.get("lighting_level", type=float),
    )
    db.session.add(sample)
    db.session.flush()

    live_count = DatasetSample.query.filter_by(word_id=word.id).count()
    word.sample_count = live_count
    db.session.commit()

    return jsonify({
        "message": "Sample saved",
        "sample_id": sample.id,
        "sample_count": live_count,
    }), 201
