from flask import Blueprint, request, jsonify
from flask_jwt_extended import (
    create_access_token, create_refresh_token,
    jwt_required, get_jwt_identity, get_jwt
)
from app import db, bcrypt
from app.models.user import User, UserProfile, TokenBlacklist
from datetime import datetime

auth_bp = Blueprint('auth', __name__)

# ── Register ──────────────────────────────────────────────
@auth_bp.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    name = data.get('name')
    email = data.get('email')
    password = data.get('password')
    role = data.get('role', 'User')

    if not name or not email or not password:
        return jsonify({'message': 'All fields are required'}), 400

    if User.query.filter_by(email=email).first():
        return jsonify({'message': 'Email already in use'}), 400

    if len(password) < 8:
        return jsonify({'message': 'Password must be at least 8 characters'}), 400

    password_hash = bcrypt.generate_password_hash(password).decode('utf-8')
    user = User(name=name, email=email, password_hash=password_hash,
                role=role, is_active=True)
    db.session.add(user)
    db.session.flush()

    profile = UserProfile(user_id=user.id, display_name=name)
    db.session.add(profile)
    db.session.commit()

    return jsonify({'message': 'Account created successfully'}), 201

# ── Login ──────────────────────────────────────────────────
@auth_bp.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')

    if not email or not password:
        return jsonify({'message': 'Email and password are required'}), 400

    user = User.query.filter_by(email=email).first()

    if not user or not bcrypt.check_password_hash(user.password_hash, password):
        return jsonify({'message': 'Invalid email or password'}), 401

    if not user.is_active:
        return jsonify({'message': 'Please verify your email first'}), 403

    access_token = create_access_token(
        identity=str(user.id),
        additional_claims={
            'role': user.role,
            'name': user.name,
            'email': user.email
        }
    )
    refresh_token = create_refresh_token(identity=str(user.id))

    return jsonify({
        'access_token': access_token,
        'refresh_token': refresh_token,
        'user': user.to_dict()
    }), 200

# ── Refresh Token ──────────────────────────────────────────
@auth_bp.route('/refresh', methods=['POST'])
@jwt_required(refresh=True)
def refresh():
    identity = get_jwt_identity()
    user = User.query.get(int(identity))
    access_token = create_access_token(
        identity=identity,
        additional_claims={
            'role': user.role,
            'name': user.name,
            'email': user.email
        }
    )
    return jsonify({'access_token': access_token}), 200

# ── Logout ─────────────────────────────────────────────────
@auth_bp.route('/logout', methods=['POST'])
@jwt_required()
def logout():
    jti = get_jwt()['jti']
    blacklisted = TokenBlacklist(jti=jti)
    db.session.add(blacklisted)
    db.session.commit()
    return jsonify({'message': 'Logged out successfully'}), 200

# ── Get Profile ────────────────────────────────────────────
@auth_bp.route('/profile', methods=['GET'])
@jwt_required()
def get_profile():
    identity = get_jwt_identity()
    user = User.query.get(int(identity))
    if not user:
        return jsonify({'message': 'User not found'}), 404
    return jsonify(user.to_dict()), 200

# ── Update Profile ─────────────────────────────────────────
@auth_bp.route('/profile', methods=['PUT'])
@jwt_required()
def update_profile():
    identity = get_jwt_identity()
    user = User.query.get(int(identity))
    if not user:
        return jsonify({'message': 'User not found'}), 404

    data = request.get_json()
    if 'name' in data:
        user.name = data['name']
        if user.profile:
            user.profile.display_name = data['name']
            user.profile.updated_at = datetime.utcnow()

    db.session.commit()
    return jsonify({'message': 'Profile updated', 'user': user.to_dict()}), 200

# ── Get All Users (Admin) ──────────────────────────────────
@auth_bp.route('/users', methods=['GET'])
@jwt_required()
def get_users():
    claims = get_jwt()
    if claims.get('role') != 'Admin':
        return jsonify({'message': 'Admin access required'}), 403

    users = User.query.all()
    return jsonify([u.to_dict() for u in users]), 200