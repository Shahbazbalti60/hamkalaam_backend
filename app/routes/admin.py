"""
app/routes/admin.py

Admin user-management endpoints for HamKalaam.
Every endpoint in this file is admin-only.

Endpoints:
    GET /api/admin/users              -> list all users
    PUT /api/admin/user/<id>/role     -> change a user's role
    PUT /api/admin/user/<id>/status   -> activate/deactivate a user
    GET /api/admin/system-stats       -> counts, 7-day activity, top words, health
    GET /api/admin/model-status       -> recognition model details
    POST /api/admin/model/reload      -> reload the model from disk
    POST   /api/admin/expression-samples          -> save face vectors for training
    GET    /api/admin/expression-samples/summary  -> how many per class
    DELETE /api/admin/expression-samples/<label>  -> remove one class' samples
"""

import importlib.util
import json
import os
import threading
from collections import Counter
from datetime import datetime, timedelta

from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from marshmallow import Schema, fields, ValidationError, validate

from app import db
from app.models.user import (
    User, Session, SessionSentence, DatasetWord, DatasetSample,
)

admin_bp = Blueprint("admin", __name__, url_prefix="/api/admin")

# Only these two role values are accepted. Keep in sync with whatever
# your auth.py / frontend actually uses for role names.
ALLOWED_ROLES = ["User", "Admin"]


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
    if (user.role or "").lower() != "admin":
        return jsonify({"error": "Admin access required"}), 403
    return None


# ─────────────────────────────────────────────────────────────
# Validation schemas
# ─────────────────────────────────────────────────────────────

class RoleSchema(Schema):
    role = fields.Str(required=True, validate=validate.OneOf(ALLOWED_ROLES))


class StatusSchema(Schema):
    is_active = fields.Bool(required=True)


role_schema = RoleSchema()
status_schema = StatusSchema()


# ─────────────────────────────────────────────────────────────
# GET /api/admin/users
# ─────────────────────────────────────────────────────────────

@admin_bp.route("/users", methods=["GET"])
@jwt_required()
def list_users():
    if err := _require_admin():
        return err

    users = User.query.order_by(User.created_at.desc()).all()
    return jsonify({
        "users": [u.to_dict() for u in users],
        "total": len(users),
    }), 200


# ─────────────────────────────────────────────────────────────
# PUT /api/admin/user/<id>/role
# ─────────────────────────────────────────────────────────────

@admin_bp.route("/user/<int:user_id>/role", methods=["PUT"])
@jwt_required()
def update_role(user_id):
    if err := _require_admin():
        return err

    current = _current_user()
    target = db.session.get(User, user_id)
    if target is None:
        return jsonify({"error": "User not found"}), 404

    json_data = request.get_json(silent=True) or {}
    try:
        data = role_schema.load(json_data)
    except ValidationError as e:
        return jsonify({"error": "Validation failed", "details": e.messages}), 400

    # Prevent an admin from demoting themselves and getting locked out —
    # they'd need another admin to change it back.
    if target.id == current.id and data["role"] != current.role:
        return jsonify({"error": "You cannot change your own role"}), 400

    target.role = data["role"]
    db.session.commit()

    return jsonify({
        "message": f"Role updated to {target.role}",
        "user": target.to_dict(),
    }), 200


# ─────────────────────────────────────────────────────────────
# PUT /api/admin/user/<id>/status
# ─────────────────────────────────────────────────────────────

@admin_bp.route("/user/<int:user_id>/status", methods=["PUT"])
@jwt_required()
def update_status(user_id):
    if err := _require_admin():
        return err

    current = _current_user()
    target = db.session.get(User, user_id)
    if target is None:
        return jsonify({"error": "User not found"}), 404

    json_data = request.get_json(silent=True) or {}
    try:
        data = status_schema.load(json_data)
    except ValidationError as e:
        return jsonify({"error": "Validation failed", "details": e.messages}), 400

    # Prevent an admin from deactivating their own account and locking
    # themselves out.
    if target.id == current.id and data["is_active"] is False:
        return jsonify({"error": "You cannot deactivate your own account"}), 400

    target.is_active = data["is_active"]
    db.session.commit()

    status_word = "activated" if target.is_active else "deactivated"
    return jsonify({
        "message": f"User {status_word}",
        "user": target.to_dict(),
    }), 200


# ─────────────────────────────────────────────────────────────
# GET /api/admin/system-stats?tz=<minutes east of UTC>
# ─────────────────────────────────────────────────────────────

def _iso(dt):
    """UTC datetime -> ISO string ending in 'Z'."""
    return dt.isoformat() + "Z" if dt else None


@admin_bp.route("/system-stats", methods=["GET"])
@jwt_required()
def system_stats():
    if err := _require_admin():
        return err

    # The app sends its own UTC offset (Pakistan = 300) so "days" match
    # the admin's local calendar day.
    tz = request.args.get("tz", default=0, type=int) or 0
    tz = max(-840, min(840, tz))
    offset = timedelta(minutes=tz)

    users_total = User.query.count()
    users_active = User.query.filter_by(is_active=True).count()
    admins = User.query.filter(db.func.lower(User.role) == "admin").count()

    # Sentences per local day for the last 7 days
    today_local = (datetime.utcnow() + offset).date()
    days = [today_local - timedelta(days=i) for i in range(6, -1, -1)]
    start_utc = datetime.combine(days[0], datetime.min.time()) - offset
    counts = {d: 0 for d in days}
    rows = (
        SessionSentence.query
        .filter(SessionSentence.generated_at >= start_utc)
        .with_entities(SessionSentence.generated_at)
        .all()
    )
    for (ts,) in rows:
        if ts is None:
            continue
        d = (ts + offset).date()
        if d in counts:
            counts[d] += 1
    per_day = [{"date": d.isoformat(), "count": counts[d]} for d in days]

    # Most used words (from the latest 2000 sentences)
    word_counter = Counter()
    recent = (
        SessionSentence.query
        .order_by(SessionSentence.id.desc())
        .limit(2000)
        .with_entities(SessionSentence.word_sequence)
        .all()
    )
    for (seq,) in recent:
        for w in (seq or "").split(","):
            w = w.strip()
            if w:
                word_counter[w] += 1
    top_words = [{"word": w, "count": c} for w, c in word_counter.most_common(8)]

    # Health checks (never expose secrets, only yes/no)
    try:
        db.session.execute(db.text("SELECT 1"))
        db_ok = True
    except Exception:
        db_ok = False

    model_loaded = False
    try:
        from app.routes.recognition import _sequence_classifier
        model_loaded = bool(_sequence_classifier.is_loaded)
    except Exception:
        pass

    mail_ok = False
    verification_on = False
    try:
        from app.routes.account import _mail_is_configured, verification_required
        mail_ok = bool(_mail_is_configured())
        verification_on = bool(verification_required())
    except Exception:
        pass

    return jsonify({
        "users": {"total": users_total, "active": users_active, "admins": admins},
        "sessions_total": Session.query.count(),
        "sentences_total": SessionSentence.query.count(),
        "dataset": {
            "words": DatasetWord.query.count(),
            "samples": DatasetSample.query.count(),
        },
        "sentences_per_day": per_day,
        "top_words": top_words,
        "health": {
            "database": db_ok,
            "model_loaded": model_loaded,
            "voice_configured": bool((os.environ.get("ELEVENLABS_API_KEY") or "").strip()),
            "email_configured": mail_ok,
            "email_verification_required": verification_on,
        },
    }), 200


# ─────────────────────────────────────────────────────────────
# GET /api/admin/model-status
# ─────────────────────────────────────────────────────────────

def _get_classifier():
    from app.routes.recognition import _sequence_classifier
    return _sequence_classifier


@admin_bp.route("/model-status", methods=["GET"])
@jwt_required()
def model_status():
    if err := _require_admin():
        return err

    from app.ai import sequence_classifier as sc
    try:
        clf = _get_classifier()
    except Exception as exc:
        return jsonify({"error": f"Recognition module not available: {exc}"}), 503

    model_dir = clf._model_dir
    model_path = os.path.join(model_dir, sc.MODEL_FILE)
    labels_path = os.path.join(model_dir, sc.LABELS_FILE)
    results_path = os.path.join(model_dir, "results.json")

    words = []
    if os.path.isfile(labels_path):
        try:
            with open(labels_path, "r", encoding="utf-8") as f:
                words = json.load(f).get("words", [])
        except Exception:
            words = []

    model_file = None
    if os.path.isfile(model_path):
        st = os.stat(model_path)
        model_file = {
            "name": sc.MODEL_FILE,
            "size_mb": round(st.st_size / (1024 * 1024), 2),
            "modified_at": _iso(datetime.utcfromtimestamp(st.st_mtime)),
        }

    # Optional: results.json (accuracy numbers saved by the Colab notebook)
    results = None
    if os.path.isfile(results_path) and os.path.getsize(results_path) < 100_000:
        try:
            with open(results_path, "r", encoding="utf-8") as f:
                results = json.load(f)
        except Exception:
            results = None

    parameters = None
    if clf.is_loaded:
        try:
            parameters = int(clf._model.count_params())
        except Exception:
            parameters = None

    # Facial-expression model (SRDS FR-21): present only after the face patch
    face = None
    try:
        from app.routes.recognition import _face_classifier as fc
        face_results = None
        rp = os.path.join(fc._model_dir, "face_results.json")
        if os.path.isfile(rp) and os.path.getsize(rp) < 100_000:
            with open(rp, "r", encoding="utf-8") as f:
                face_results = json.load(f)
        face = {
            "trained": bool(fc.files_present()),
            "loaded": bool(fc.is_loaded),
            "classes": fc.classes,
            "results": face_results,
        }
    except Exception:
        face = None

    return jsonify({
        "face": face,
        "loaded": bool(clf.is_loaded),
        "model_file": model_file,
        "labels_found": os.path.isfile(labels_path),
        "word_count": len(words),
        "words": words,
        "parameters": parameters,
        "tensorflow_installed": importlib.util.find_spec("tensorflow") is not None,
        "sequence_length": clf.SEQ_LEN,
        "min_hands_ratio": clf.MIN_HANDS_RATIO,
        "confident_threshold": clf.CONFIDENT,
        "results": results,
    }), 200


# ─────────────────────────────────────────────────────────────
# POST /api/admin/model/reload
# ─────────────────────────────────────────────────────────────

@admin_bp.route("/model/reload", methods=["POST"])
@jwt_required()
def model_reload():
    """Loads the model files from disk again (e.g. after copying a newly
    trained psl_lstm.keras). The running model is kept if loading fails."""
    if err := _require_admin():
        return err

    from app.ai.sequence_classifier import SequenceClassifier, ModelUnavailable
    try:
        clf = _get_classifier()
    except Exception as exc:
        return jsonify({"error": f"Recognition module not available: {exc}"}), 503

    try:
        fresh = SequenceClassifier(model_dir=clf._model_dir)
        fresh.load()
    except ModelUnavailable as exc:
        return jsonify({"error": str(exc)}), 503
    except Exception as exc:
        return jsonify({"error": f"Could not load the model: {exc}"}), 500

    with clf._lock:
        clf._model = fresh._model
        clf._words = fresh._words

    return jsonify({
        "message": "Model reloaded",
        "loaded": True,
        "word_count": len(clf.words),
    }), 200


# ─────────────────────────────────────────────────────────────
# Facial-expression training samples (SRDS FR-20 / FR-21)
# Admins record themselves making each expression; the app sends the
# 68 face landmarks (136 numbers per frame). They are appended to
#   uploads/expression_samples/samples.jsonl
# and used by the Colab notebook to train the expression CNN.
# ─────────────────────────────────────────────────────────────

EXPRESSION_CLASSES = ["Question", "Negation", "Affirmation", "Neutral", "Happy", "Sad"]
_BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EXPRESSION_DIR = os.path.join(_BACKEND_ROOT, "uploads", "expression_samples")
EXPRESSION_FILE = os.path.join(EXPRESSION_DIR, "samples.jsonl")
_expr_lock = threading.Lock()
MAX_VECTORS_PER_UPLOAD = 60


def _expression_counts():
    counts = {c: 0 for c in EXPRESSION_CLASSES}
    recs = {c: set() for c in EXPRESSION_CLASSES}
    if os.path.isfile(EXPRESSION_FILE):
        with open(EXPRESSION_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                label = row.get("label")
                if label in counts:
                    counts[label] += 1
                    recs[label].add(row.get("rec"))
    return counts, {c: len(v) for c, v in recs.items()}


@admin_bp.route("/expression-samples", methods=["POST"])
@jwt_required()
def save_expression_samples():
    if err := _require_admin():
        return err

    data = request.get_json(silent=True) or {}
    label = data.get("label")
    vectors = data.get("vectors")
    rec_id = str(data.get("rec_id") or "")[:64]
    if label not in EXPRESSION_CLASSES:
        return jsonify({"error": f"label must be one of {EXPRESSION_CLASSES}"}), 400
    if not isinstance(vectors, list) or not (1 <= len(vectors) <= MAX_VECTORS_PER_UPLOAD):
        return jsonify({"error": f"send 1 to {MAX_VECTORS_PER_UPLOAD} face vectors"}), 400

    clean = []
    for v in vectors:
        if (not isinstance(v, list) or len(v) != 136
                or not all(isinstance(x, (int, float)) and abs(x) < 1e6 for x in v)):
            return jsonify({"error": "each face vector must be 136 numbers"}), 400
        clean.append([round(float(x), 2) for x in v])

    user = _current_user()
    os.makedirs(EXPRESSION_DIR, exist_ok=True)
    stamp = datetime.utcnow().isoformat() + "Z"
    with _expr_lock:
        with open(EXPRESSION_FILE, "a", encoding="utf-8") as f:
            for v in clean:
                f.write(json.dumps({
                    "label": label, "rec": rec_id, "signer": user.id if user else None,
                    "t": stamp, "points": v,
                }) + "\n")
        counts, recs = _expression_counts()
    return jsonify({"saved": len(clean), "counts": counts, "recordings": recs}), 201


@admin_bp.route("/expression-samples/summary", methods=["GET"])
@jwt_required()
def expression_samples_summary():
    if err := _require_admin():
        return err
    with _expr_lock:
        counts, recs = _expression_counts()
    return jsonify({"counts": counts, "recordings": recs,
                    "total": sum(counts.values()),
                    "file": "uploads/expression_samples/samples.jsonl"}), 200


@admin_bp.route("/expression-samples/<label>", methods=["DELETE"])
@jwt_required()
def delete_expression_class(label):
    if err := _require_admin():
        return err
    if label not in EXPRESSION_CLASSES:
        return jsonify({"error": "Unknown class"}), 404
    removed = 0
    with _expr_lock:
        if os.path.isfile(EXPRESSION_FILE):
            kept = []
            with open(EXPRESSION_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        if json.loads(line).get("label") == label:
                            removed += 1
                            continue
                    except ValueError:
                        pass
                    kept.append(line)
            with open(EXPRESSION_FILE, "w", encoding="utf-8") as f:
                f.writelines(kept)
        counts, recs = _expression_counts()
    return jsonify({"removed": removed, "counts": counts, "recordings": recs}), 200
