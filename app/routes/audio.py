"""
app/routes/audio.py

Text-to-speech endpoint for HamKalaam. Proxies requests to ElevenLabs
so the API key stays server-side (in .env) and is never exposed to
the Flutter web client.

Endpoint:
    POST /api/audio/speak
        Input:  {text: str}
        Output: raw audio/mpeg bytes (the generated MP3)
"""

import os

import requests
from flask import Blueprint, request, jsonify, Response
from flask_jwt_extended import jwt_required
from marshmallow import Schema, fields, ValidationError, validate

audio_bp = Blueprint("audio", __name__, url_prefix="/api/audio")

ELEVENLABS_API_URL = "https://api.elevenlabs.io/v1/text-to-speech"
# Default model chosen for broad multilingual support. Urdu is not
# officially listed as a supported language for eleven_multilingual_v2,
# but the model still produces intelligible speech for Urdu script in
# practice — verified by ear during development, not by an official
# ElevenLabs language guarantee. If quality is ever an issue, consider
# testing eleven_v3 as an alternative.
ELEVENLABS_MODEL_ID = "eleven_multilingual_v2"


class SpeakSchema(Schema):
    text = fields.Str(required=True, validate=validate.Length(min=1, max=2000))


speak_schema = SpeakSchema()


@audio_bp.route("/speak", methods=["POST"])
@jwt_required()
def speak():
    api_key = os.environ.get("ELEVENLABS_API_KEY")
    voice_id = os.environ.get("ELEVENLABS_VOICE_ID")

    if not api_key or not voice_id:
        return jsonify({
            "error": "TTS is not configured on the server "
                     "(missing ELEVENLABS_API_KEY or ELEVENLABS_VOICE_ID in .env)"
        }), 500

    json_data = request.get_json(silent=True) or {}
    try:
        data = speak_schema.load(json_data)
    except ValidationError as e:
        return jsonify({"error": "Validation failed", "details": e.messages}), 400

    try:
        response = requests.post(
            f"{ELEVENLABS_API_URL}/{voice_id}",
            params={"output_format": "mp3_44100_128"},
            headers={
                "xi-api-key": api_key,
                "Content-Type": "application/json",
            },
            json={
                "text": data["text"],
                "model_id": ELEVENLABS_MODEL_ID,
            },
            timeout=30,
        )
    except requests.RequestException:
        return jsonify({"error": "Could not reach the text-to-speech service"}), 502

    if response.status_code != 200:
        # Forward a trimmed version of ElevenLabs' error so it's
        # debuggable without leaking the full raw response.
        return jsonify({
            "error": "Text-to-speech generation failed",
            "status": response.status_code,
        }), 502

    return Response(response.content, mimetype="audio/mpeg")