from datetime import datetime
from app import db

class User(db.Model):
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(20), default='User')
    is_active = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    profile = db.relationship('UserProfile', backref='user', uselist=False)
    sessions = db.relationship('Session', backref='user', lazy=True)

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'email': self.email,
            'role': self.role,
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat()
        }

class UserProfile(db.Model):
    __tablename__ = 'user_profiles'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    display_name = db.Column(db.String(100))
    profile_photo = db.Column(db.String(255))
    urdu_dialect = db.Column(db.String(50), default='Standard')
    updated_at = db.Column(db.DateTime, default=datetime.utcnow)

class Session(db.Model):
    __tablename__ = 'sessions'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    started_at = db.Column(db.DateTime, default=datetime.utcnow)
    ended_at = db.Column(db.DateTime)
    sentences = db.relationship('SessionSentence', backref='session', lazy=True)

class SessionSentence(db.Model):
    __tablename__ = 'session_sentences'
    
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey('sessions.id'), nullable=False)
    urdu_sentence = db.Column(db.Text, nullable=False)
    word_sequence = db.Column(db.String(500))
    expression_context = db.Column(db.String(50))
    generated_at = db.Column(db.DateTime, default=datetime.utcnow)

class DatasetWord(db.Model):
    __tablename__ = 'dataset_words'
    
    id = db.Column(db.Integer, primary_key=True)
    urdu_word = db.Column(db.String(100), unique=True, nullable=False)
    english_gloss = db.Column(db.String(100))
    domain = db.Column(db.String(50))
    sample_count = db.Column(db.Integer, default=0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    samples = db.relationship('DatasetSample', backref='word', lazy=True)

class DatasetSample(db.Model):
    __tablename__ = 'dataset_samples'
    
    id = db.Column(db.Integer, primary_key=True)
    word_id = db.Column(db.Integer, db.ForeignKey('dataset_words.id'), nullable=False)
    file_path = db.Column(db.String(255))
    duration_seconds = db.Column(db.Float)
    signer_id = db.Column(db.Integer, db.ForeignKey('users.id'))
    lighting_level = db.Column(db.Float)
    captured_at = db.Column(db.DateTime, default=datetime.utcnow)

class TokenBlacklist(db.Model):
    __tablename__ = 'token_blacklist'
    
    id = db.Column(db.Integer, primary_key=True)
    jti = db.Column(db.String(36), unique=True, nullable=False)
    blacklisted_at = db.Column(db.DateTime, default=datetime.utcnow)