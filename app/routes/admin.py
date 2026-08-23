"""
app/routes/admin.py

Admin user-management endpoints for HamKalaam.
Every endpoint in this file is admin-only.

Endpoints:
    GET /api/admin/users              -> list all users
    PUT /api/admin/user/<id>/role     -> change a user's role
    PUT /api/admin/user/<id>/status   -> activate/deactivate a user
"""

from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from marshmallow import Schema, fields, ValidationError, validate

from app import db
from app.models.user import User

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