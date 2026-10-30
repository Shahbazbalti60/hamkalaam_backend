"""
app/routes/account.py

Small features that sit next to auth.py:

1. Silent token renewal (fixes "Token has expired" mid-session)
       POST /api/auth/renew
   Needs a still-valid access token; returns a fresh one.

2. Password reset by e-mailed 6-digit code (FR-06)
       POST /api/auth/password/forgot   {email}
       POST /api/auth/password/reset    {email, code, new_password}

3. E-mail verification by e-mailed 6-digit code (FR-02)
       POST /api/auth/verify-email          {email, code}
       POST /api/auth/verify-email/resend   {email}
   Switched on with   REQUIRE_EMAIL_VERIFICATION=true   in .env
   (default: off, so nothing changes until the app has the verify screen).
   Existing accounts are treated as already verified.

DEV MODE: if the .env still has placeholder mail credentials
(MAIL_USERNAME empty or "your-email@gmail.com"), no e-mail is sent;
the code is printed in the Flask terminal instead so the whole flow can
still be demonstrated. Put a real Gmail address + App Password in .env
to send real e-mails.
"""

import os
import random
from datetime import datetime, timedelta, timezone

from flask import Blueprint, Response, request, jsonify
from flask_jwt_extended import (
    jwt_required, get_jwt_identity, get_jwt, create_access_token,
    verify_jwt_in_request,
)
from flask_mail import Message
from marshmallow import Schema, fields, ValidationError, validate

from app import db, bcrypt, mail
from app.models.user import User, UserProfile

account_bp = Blueprint("account", __name__, url_prefix="/api/auth")

# How long a (renewed) access token lives. Keep in sync with
# JWT_ACCESS_TOKEN_EXPIRES in config.py so login tokens last just as long.
ACCESS_TOKEN_LIFETIME = timedelta(hours=8)

RESET_CODE_MINUTES = 15
VERIFY_CODE_MINUTES = 30
RESEND_COOLDOWN_SECONDS = 60
MAX_CODE_ATTEMPTS = 5


# ─────────────────────────────────────────────────────────────
# Models (tables are created automatically by db.create_all() in
# create_app(), because this module is imported before it runs).
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


class EmailVerificationCode(db.Model):
    __tablename__ = "email_verification_codes"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    code_hash = db.Column(db.String(255), nullable=False)
    expires_at = db.Column(db.DateTime, nullable=False)
    attempts = db.Column(db.Integer, default=0)
    used = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class TokenRevocation(db.Model):
    """Every login token of this user issued BEFORE `revoked_before` is
    refused (set when the password is reset - SRDS FR-06)."""
    __tablename__ = "token_revocations"

    user_id = db.Column(
        db.Integer, db.ForeignKey("users.id", ondelete="CASCADE"), primary_key=True
    )
    revoked_before = db.Column(db.DateTime, nullable=False)


class UnverifiedUser(db.Model):
    """One row per account whose e-mail is NOT verified yet.
    No row = verified (so all accounts created before this feature
    keep working without any database change)."""
    __tablename__ = "unverified_users"

    user_id = db.Column(
        db.Integer, db.ForeignKey("users.id", ondelete="CASCADE"), primary_key=True
    )
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


# ─────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────

def _mail_is_configured():
    username = (os.environ.get("MAIL_USERNAME") or "").strip()
    return bool(username) and "your-email" not in username


def _send_mail_code(user, code, subject, intro, minutes, dev_label):
    """Returns True if the code was delivered (or printed in dev mode)."""
    if not _mail_is_configured():
        print(f"\n[DEV MODE] {dev_label} for {user.email}: {code}\n")
        return True

    try:
        msg = Message(
            subject=subject,
            recipients=[user.email],
            body=(
                f"Hello {user.name},\n\n"
                f"{intro} {code}\n"
                f"It expires in {minutes} minutes.\n\n"
                "If you did not request this, you can ignore this e-mail."
            ),
        )
        mail.send(msg)
        return True
    except Exception as exc:  # network / bad app password / etc.
        print(f"[MAIL ERROR] could not send e-mail: {exc}")
        return False


def _new_code():
    return f"{random.SystemRandom().randint(0, 999999):06d}"


def _find_user(email):
    return User.query.filter(db.func.lower(User.email) == email.lower()).first()


def revoke_all_tokens(user_id):
    """Cancels every access/refresh token issued to this user so far.
    (Call before db.session.commit().)"""
    now = datetime.utcnow()
    rec = db.session.get(TokenRevocation, user_id)
    if rec is None:
        db.session.add(TokenRevocation(user_id=user_id, revoked_before=now))
    else:
        rec.revoked_before = now


@account_bp.before_app_request
def _reject_revoked_tokens():
    """Runs before every request. If the request carries a login token that
    was issued before the user's password was reset, answer 401 so the app
    sends the person back to the login screen."""
    if request.method == "OPTIONS":
        return None
    if not request.headers.get("Authorization", "").startswith("Bearer "):
        return None
    try:
        verify_jwt_in_request(optional=True)
        claims = get_jwt()
        identity = get_jwt_identity()
        if not claims or identity is None:
            return None
        record = db.session.get(TokenRevocation, int(identity))
    except Exception:
        return None  # invalid/expired tokens are handled by the route itself
    if record is None or claims.get("iat") is None:
        return None
    cutoff = record.revoked_before.replace(tzinfo=timezone.utc).timestamp()
    if claims["iat"] < cutoff:
        return jsonify({"msg": "Token has been revoked. Please log in again."}), 401
    return None


# ─────────────────────────────────────────────────────────────
# 1) Token renewal
# ─────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────
# 1b) Profile extras: photo and Urdu voice dialect (FR-07)
#     (the name is edited through the existing PUT /api/auth/profile)
# ─────────────────────────────────────────────────────────────

DIALECTS = ("Standard", "Punjabi")  # Standard Urdu / Punjabi-accented Urdu
PHOTO_MAX_BYTES = 2 * 1024 * 1024    # 2 MB
_BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PHOTO_DIR = os.path.join(_BACKEND_ROOT, "uploads", "profile_photos")


def _sniff_image(data):
    """'png' / 'jpg' from the file's first bytes (never trust the file name)."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if data[:3] == b"\xff\xd8\xff":
        return "jpg"
    return None


def _profile_for(user):
    profile = user.profile
    if profile is None:
        profile = UserProfile(user_id=user.id, display_name=user.name)
        db.session.add(profile)
        db.session.flush()
    return profile


def _photo_file(profile):
    """Absolute path of the stored photo, or None. Only files inside the
    profile_photos folder are ever served."""
    if not profile or not profile.profile_photo:
        return None
    full = os.path.abspath(os.path.join(_BACKEND_ROOT, profile.profile_photo))
    if not full.startswith(os.path.abspath(PHOTO_DIR) + os.sep) or not os.path.isfile(full):
        return None
    return full


def _remove_photo_files(user_id):
    for ext in ("png", "jpg"):
        path = os.path.join(PHOTO_DIR, f"user_{user_id}.{ext}")
        if os.path.isfile(path):
            try:
                os.remove(path)
            except OSError:
                pass


@account_bp.route("/profile/extra", methods=["GET"])
@jwt_required()
def profile_extra():
    user = db.session.get(User, int(get_jwt_identity()))
    if user is None:
        return jsonify({"error": "User not found"}), 404
    profile = user.profile
    return jsonify({
        "urdu_dialect": (profile.urdu_dialect if profile and profile.urdu_dialect else "Standard"),
        "has_photo": _photo_file(profile) is not None,
    }), 200


class DialectSchema(Schema):
    urdu_dialect = fields.Str(required=True, validate=validate.OneOf(DIALECTS))


@account_bp.route("/profile/dialect", methods=["PUT"])
@jwt_required()
def set_dialect():
    user = db.session.get(User, int(get_jwt_identity()))
    if user is None:
        return jsonify({"error": "User not found"}), 404
    try:
        data = DialectSchema().load(request.get_json(silent=True) or {})
    except ValidationError as e:
        return jsonify({"error": "Validation failed", "details": e.messages}), 400

    profile = _profile_for(user)
    profile.urdu_dialect = data["urdu_dialect"]
    profile.updated_at = datetime.utcnow()
    db.session.commit()
    return jsonify({"message": "Dialect saved", "urdu_dialect": profile.urdu_dialect}), 200


@account_bp.route("/profile/photo", methods=["POST"])
@jwt_required()
def upload_photo():
    user = db.session.get(User, int(get_jwt_identity()))
    if user is None:
        return jsonify({"error": "User not found"}), 404

    file = request.files.get("photo")
    if file is None:
        return jsonify({"error": "No photo was sent"}), 400

    data = file.read(PHOTO_MAX_BYTES + 1)
    if len(data) > PHOTO_MAX_BYTES:
        return jsonify({"error": "Photo must be 2 MB or smaller"}), 413
    kind = _sniff_image(data)
    if kind is None:
        return jsonify({"error": "Only JPEG or PNG photos are allowed"}), 400

    os.makedirs(PHOTO_DIR, exist_ok=True)
    _remove_photo_files(user.id)  # one photo per user
    with open(os.path.join(PHOTO_DIR, f"user_{user.id}.{kind}"), "wb") as out:
        out.write(data)

    profile = _profile_for(user)
    profile.profile_photo = f"uploads/profile_photos/user_{user.id}.{kind}"
    profile.updated_at = datetime.utcnow()
    db.session.commit()
    return jsonify({"message": "Photo saved"}), 201


@account_bp.route("/profile/photo", methods=["GET"])
@jwt_required()
def get_photo():
    user = db.session.get(User, int(get_jwt_identity()))
    if user is None:
        return jsonify({"error": "User not found"}), 404
    path = _photo_file(user.profile)
    if path is None:
        return jsonify({"error": "No photo"}), 404
    with open(path, "rb") as f:
        body = f.read()
    mimetype = "image/png" if path.endswith(".png") else "image/jpeg"
    return Response(body, mimetype=mimetype, headers={"Cache-Control": "no-store"})


@account_bp.route("/profile/photo", methods=["DELETE"])
@jwt_required()
def delete_photo():
    user = db.session.get(User, int(get_jwt_identity()))
    if user is None:
        return jsonify({"error": "User not found"}), 404
    _remove_photo_files(user.id)
    if user.profile is not None:
        user.profile.profile_photo = None
        user.profile.updated_at = datetime.utcnow()
    db.session.commit()
    return jsonify({"message": "Photo removed"}), 200


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


def _send_code(user, code):
    return _send_mail_code(
        user, code,
        subject="HamKalaam password reset code",
        intro="Your HamKalaam password reset code is:",
        minutes=RESET_CODE_MINUTES,
        dev_label="Password reset code",
    )


@account_bp.route("/password/forgot", methods=["POST"])
def forgot_password():
    try:
        data = ForgotSchema().load(request.get_json(silent=True) or {})
    except ValidationError as e:
        return jsonify({"error": "Validation failed", "details": e.messages}), 400

    generic = {"message": "If that e-mail is registered, a reset code has been sent."}

    user = _find_user(data["email"])
    if user is None:
        # Same answer either way so the endpoint can't be used to
        # find out which e-mails are registered.
        return jsonify(generic), 200

    # Invalidate any earlier unused codes for this user.
    PasswordResetCode.query.filter_by(user_id=user.id, used=False).update({"used": True})

    code = _new_code()
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

    user = _find_user(data["email"])
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
    revoke_all_tokens(user.id)  # FR-06: older logins stop working
    db.session.commit()

    return jsonify({"message": "Password updated. You can now log in."}), 200


# ─────────────────────────────────────────────────────────────
# 3) E-mail verification (FR-02)
# ─────────────────────────────────────────────────────────────

def verification_required():
    """True when the .env switch REQUIRE_EMAIL_VERIFICATION is on."""
    return (os.environ.get("REQUIRE_EMAIL_VERIFICATION") or "").strip().lower() in (
        "1", "true", "yes", "on")


def is_unverified(user):
    return db.session.get(UnverifiedUser, user.id) is not None


def issue_verification_code(user):
    """Marks the user as unverified (if not already), creates a fresh
    6-digit code (older ones stop working) and sends it.
    Returns True if the code was delivered (or printed in dev mode)."""
    if db.session.get(UnverifiedUser, user.id) is None:
        db.session.add(UnverifiedUser(user_id=user.id))

    EmailVerificationCode.query.filter_by(user_id=user.id, used=False).update({"used": True})
    code = _new_code()
    db.session.add(EmailVerificationCode(
        user_id=user.id,
        code_hash=bcrypt.generate_password_hash(code).decode("utf-8"),
        expires_at=datetime.utcnow() + timedelta(minutes=VERIFY_CODE_MINUTES),
    ))
    db.session.commit()

    return _send_mail_code(
        user, code,
        subject="HamKalaam e-mail verification code",
        intro="Your HamKalaam verification code is:",
        minutes=VERIFY_CODE_MINUTES,
        dev_label="E-mail verification code",
    )


class VerifySchema(Schema):
    email = fields.Email(required=True)
    code = fields.Str(required=True, validate=validate.Length(equal=6))


@account_bp.route("/verify-email", methods=["POST"])
def verify_email():
    try:
        data = VerifySchema().load(request.get_json(silent=True) or {})
    except ValidationError as e:
        return jsonify({"error": "Validation failed", "details": e.messages}), 400

    bad = jsonify({"error": "Invalid or expired code"}), 400

    user = _find_user(data["email"])
    if user is None:
        return bad

    pending = db.session.get(UnverifiedUser, user.id)
    if pending is None:
        # Already verified (or an older account): nothing to do.
        return jsonify({"message": "E-mail already verified. You can log in."}), 200

    record = (
        EmailVerificationCode.query
        .filter_by(user_id=user.id, used=False)
        .order_by(EmailVerificationCode.created_at.desc())
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

    record.used = True
    db.session.delete(pending)
    db.session.commit()
    return jsonify({"message": "E-mail verified. You can now log in."}), 200


@account_bp.route("/verify-email/resend", methods=["POST"])
def resend_verification():
    try:
        data = ForgotSchema().load(request.get_json(silent=True) or {})
    except ValidationError as e:
        return jsonify({"error": "Validation failed", "details": e.messages}), 400

    generic = {"message": "If that account needs verification, a new code has been sent."}

    user = _find_user(data["email"])
    if user is None or db.session.get(UnverifiedUser, user.id) is None:
        return jsonify(generic), 200

    last = (
        EmailVerificationCode.query
        .filter_by(user_id=user.id)
        .order_by(EmailVerificationCode.created_at.desc())
        .first()
    )
    if last and last.created_at:
        wait = RESEND_COOLDOWN_SECONDS - (datetime.utcnow() - last.created_at).total_seconds()
        if wait > 0:
            return jsonify({
                "error": f"Please wait {int(wait) + 1} seconds before asking for a new code.",
                "retry_after": int(wait) + 1,
            }), 429

    if not issue_verification_code(user):
        return jsonify({"error": "Could not send the e-mail. Please try again later."}), 502
    return jsonify(generic), 200
