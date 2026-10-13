"""
app/routes/account.py

Two small features that sit next to auth.py (kept in a separate file
so the existing, tested auth.py is not touched):

1. Silent token renewal (fixes "Token has expired" mid-session)
       POST /api/auth/renew
   Needs a still-valid access token; returns a fresh one.

2. Password reset by e-mailed 6-digit code (FR-06)
       POST /api/auth/password/forgot   {email}
       POST /api/auth/password/reset    {email, code, new_password}

DEV MODE: if the .env still has placeholder mail credentials
(MAIL_USERNAME empty or "your-email@gmail.com"), no e-mail is sent;
the code is printed in the Flask terminal instead so the whole flow can
still be demonstrated. Put a real Gmail address + App Password in .env
to send real e-mails.
"""

import os
import random
from datetime import datetime, timedelta

from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity, create_access_token
from flask_mail import Message
from marshmallow import Schema, fields, ValidationError, validate

from app import db, bcrypt, mail
from app.models.user import User

account_bp = Blueprint("account", __name__, url_prefix="/api/auth")

# How long a (renewed) access token lives. Keep in sync with
# JWT_ACCESS_TOKEN_EXPIRES in config.py so login tokens last just as long.
ACCESS_TOKEN_LIFETIME = timedelta(hours=8)

RESET_CODE_MINUTES = 15
MAX_CODE_ATTEMPTS = 5


# ─────────────────────────────────────────────────────────────
# Model for reset codes (table is created automatically by
# db.create_all() in create_app(), because this module is imported
# before create_all runs).
# ─────────────────────────────────────────────────────────────

class PasswordResetCode(db.Model):
    __tablename__ = "password_reset_codes"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    code_hash = db.Column(db.String(255), nullable=False)
    expires_at = db.Column(db.DateTime, nullable=False)
    attempts = db.Column(db.Integer, default=0)
    used = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


# ─────────────────────────────────────────────────────────────
# 1) Token renewal
# ─────────────────────────────────────────────────────────────

@account_bp.route("/renew", methods=["POST"])
@jwt_required()
def renew_token():
    user = db.session.get(User, int(get_jwt_identity()))
    if user is None:
        return jsonify({"error": "User not found"}), 404
    if not user.is_active:
        return jsonify({"error": "Account is deactivated"}), 403

    # Claims are rebuilt from the database (not copied from the old
    # token) so a changed name/role is picked up, and a deactivated
    # account can no longer keep itself alive.
    token = create_access_token(
        identity=str(user.id),
        additional_claims={
            "role": user.role,
            "name": user.name,
            "email": user.email,
        },
        expires_delta=ACCESS_TOKEN_LIFETIME,
    )
    return jsonify({"access_token": token}), 200


# ─────────────────────────────────────────────────────────────
# 2) Password reset
# ─────────────────────────────────────────────────────────────

class ForgotSchema(Schema):
    email = fields.Email(required=True)


class ResetSchema(Schema):
    email = fields.Email(required=True)
    code = fields.Str(required=True, validate=validate.Length(equal=6))
    new_password = fields.Str(required=True, validate=validate.Length(min=8, max=128))


def _mail_is_configured():
    username = (os.environ.get("MAIL_USERNAME") or "").strip()
    return bool(username) and "your-email" not in username


def _send_code(user, code):
    """Returns True if the code was delivered (or printed in dev mode)."""
    if not _mail_is_configured():
        print(f"\n[DEV MODE] Password reset code for {user.email}: {code}\n")
        return True

    try:
        msg = Message(
            subject="HamKalaam password reset code",
            recipients=[user.email],
            body=(
                f"Hello {user.name},\n\n"
                f"Your HamKalaam password reset code is: {code}\n"
                f"It expires in {RESET_CODE_MINUTES} minutes.\n\n"
                "If you did not request this, you can ignore this e-mail."
            ),
        )
        mail.send(msg)
        return True
    except Exception as exc:  # network / bad app password / etc.
        print(f"[MAIL ERROR] could not send reset e-mail: {exc}")
        return False


@account_bp.route("/password/forgot", methods=["POST"])
def forgot_password():
    try:
        data = ForgotSchema().load(request.get_json(silent=True) or {})
    except ValidationError as e:
        return jsonify({"error": "Validation failed", "details": e.messages}), 400

    generic = {"message": "If that e-mail is registered, a reset code has been sent."}

    user = User.query.filter(db.func.lower(User.email) == data["email"].lower()).first()
    if user is None:
        # Same answer either way so the endpoint can't be used to
        # find out which e-mails are registered.
        return jsonify(generic), 200

    # Invalidate any earlier unused codes for this user.
    PasswordResetCode.query.filter_by(user_id=user.id, used=False).update({"used": True})

    code = f"{random.SystemRandom().randint(0, 999999):06d}"
    db.session.add(PasswordResetCode(
        user_id=user.id,
        code_hash=bcrypt.generate_password_hash(code).decode("utf-8"),
        expires_at=datetime.utcnow() + timedelta(minutes=RESET_CODE_MINUTES),
    ))
    db.session.commit()

    if not _send_code(user, code):
        return jsonify({"error": "Could not send the e-mail. Please try again later."}), 502

    return jsonify(generic), 200


@account_bp.route("/password/reset", methods=["POST"])
def reset_password():
    try:
        data = ResetSchema().load(request.get_json(silent=True) or {})
    except ValidationError as e:
        return jsonify({"error": "Validation failed", "details": e.messages}), 400

    bad = jsonify({"error": "Invalid or expired code"}), 400

    user = User.query.filter(db.func.lower(User.email) == data["email"].lower()).first()
    if user is None:
        return bad

    record = (
        PasswordResetCode.query
        .filter_by(user_id=user.id, used=False)
        .order_by(PasswordResetCode.created_at.desc())
        .first()
    )
    if (
        record is None
        or record.expires_at < datetime.utcnow()
        or (record.attempts or 0) >= MAX_CODE_ATTEMPTS
    ):
        return bad

    if not bcrypt.check_password_hash(record.code_hash, data["code"]):
        record.attempts = (record.attempts or 0) + 1
        db.session.commit()
        return bad

    user.password_hash = bcrypt.generate_password_hash(data["new_password"]).decode("utf-8")
    record.used = True
    db.session.commit()

    return jsonify({"message": "Password updated. You can now log in."}), 200
