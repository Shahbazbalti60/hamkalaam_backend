r"""Makes recognition.py use app/ai/word_classifier.build_word_classifier()
(so WORD_MODEL=lstm|forest in .env chooses the word model).

Run ONCE from E:\Projects\hamkalaam_backend:   python patch_word_model.py
Changes one line (backup: recognition.py.bak)."""
import re
import shutil
import sys

P = "app/routes/recognition.py"
with open(P, "r", encoding="utf-8", newline="") as f:
    raw = f.read()
if "build_word_classifier" in raw:
    sys.exit("Already patched.")

pattern = re.compile(r"^(?P<ind>[ \t]*)_sequence_classifier\s*=\s*SequenceClassifier\(\s*\)[ \t]*$", re.M)
matches = pattern.findall(raw)
if len(matches) != 1:
    sys.exit("Could not find the line '_sequence_classifier = SequenceClassifier()' exactly once - "
             "nothing changed. Send me the top 30 lines of recognition.py.")

new = pattern.sub(
    "from app.ai.word_classifier import build_word_classifier\n"
    "_sequence_classifier = build_word_classifier()", raw, count=1)
shutil.copy(P, P + ".bak")
with open(P, "w", encoding="utf-8", newline="") as f:
    f.write(new)
print("OK: recognition.py now uses build_word_classifier(). Backup:", P + ".bak")
