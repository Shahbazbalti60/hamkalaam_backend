r"""Adds the face-expression endpoints to app/routes/recognition.py:
    POST /api/recognition/expression          {landmarks: [[136 numbers], ...]}
    GET  /api/recognition/expression-status

Run ONCE from E:\Projects\hamkalaam_backend:   python patch_recognition_expression.py
It only APPENDS to the end of the file (backup: recognition.py.bak)."""
import shutil
import sys

P = "app/routes/recognition.py"
MARK = "# ---- face expression (SRDS FR-20 to FR-22) ----"

ADD = '''

''' + MARK + '''
from app.ai.face_classifier import FaceClassifier, FaceModelUnavailable

_face_classifier = FaceClassifier()


@recognition_bp.route("/expression", methods=["POST"])
@jwt_required()
def classify_expression():
    json_data = request.get_json(silent=True) or {}
    vectors = json_data.get("landmarks")
    try:
        results = _face_classifier.predict(vectors)
    except FaceModelUnavailable as exc:
        return jsonify({"error": str(exc), "model_ready": False}), 503
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"results": results, "model_ready": True}), 200


@recognition_bp.route("/expression-status", methods=["GET"])
@jwt_required()
def expression_status():
    if not _face_classifier.is_loaded:
        if not _face_classifier.files_present():
            return jsonify({"ready": False, "reason": "not_trained"}), 200
        try:
            _face_classifier.load()
        except FaceModelUnavailable as exc:
            return jsonify({"ready": False, "reason": str(exc)}), 200
    return jsonify({"ready": True, "classes": _face_classifier.classes}), 200
'''

with open(P, "r", encoding="utf-8", newline="") as f:
    raw = f.read()
if MARK in raw:
    sys.exit("Already patched.")
if "recognition_bp" not in raw or "jwt_required" not in raw:
    sys.exit("This does not look like recognition.py - nothing changed.")
crlf = "\r\n" in raw
add = ADD.replace("\n", "\r\n") if crlf else ADD
shutil.copy(P, P + ".bak")
with open(P, "w", encoding="utf-8", newline="") as f:
    f.write(raw.rstrip("\r\n") + "\r\n" + add if crlf else raw.rstrip("\n") + "\n" + add)
print("OK: face expression endpoints added. Backup:", P + ".bak")
