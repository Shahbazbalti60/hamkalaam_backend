r"""Removes the LSTM from the running app. The Random Forest becomes the only word model.

Run ONCE from E:\Projects\hamkalaam_backend (venv active):   python remove_lstm.py

What it does
  1. recognition.py: the import of the LSTM module is replaced by the forest module's
     ModelUnavailable (backup: recognition.py.bak).
  2. Looks for any other file that still uses the LSTM module. If it finds one it
     STOPS and lists it - nothing is moved.
  3. Moves the LSTM files out of the app into ..\lstm_archive\ (nothing is deleted, so the
     comparison in the report can still be reproduced):
        app/ai/sequence_classifier.py, app/ai/model/psl_lstm.keras, app/ai/model/labels.json
     (results.json stays: it holds the comparison numbers shown in Model Management.)
"""
import os
import re
import shutil
import sys

REC = "app/routes/recognition.py"
ARCHIVE = os.path.join("..", "lstm_archive")
MOVE = [
    "app/ai/sequence_classifier.py",
    "app/ai/model/psl_lstm.keras",
    "app/ai/model/labels.json",
]

if not os.path.isfile("app/ai/forest_classifier.py"):
    sys.exit("app/ai/forest_classifier.py not found - install the Random Forest zip first.")
if not os.path.isfile("app/ai/model/psl_forest.joblib") or not os.path.isfile("app/ai/model/forest_labels.json"):
    sys.exit("The forest files are not in app/ai/model/ yet - copy them first, then run this again.")

with open(REC, "r", encoding="utf-8", newline="") as f:
    raw = f.read()

# 1) recognition.py: the LSTM import line(s) -> the forest module's ModelUnavailable
pat = re.compile(r"^[ \t]*from app\.ai\.sequence_classifier import [^\n]*\n", re.M)
n = len(pat.findall(raw))
new = raw
if n:
    new = pat.sub("", raw)
    if "from app.ai.forest_classifier import ModelUnavailable" not in new:
        # put the replacement import where the old one was (before the first non-import use)
        m = re.search(r"^_sequence_classifier|^from app\.ai\.word_classifier import", new, re.M)
        at = m.start() if m else 0
        new = new[:at] + "from app.ai.forest_classifier import ModelUnavailable\n" + new[at:]

# 2) any other user of the LSTM module?
USES_LSTM = re.compile(
    r"(from\s+app\.ai\.sequence_classifier\b|from\s+\.sequence_classifier\b|"
    r"from\s+app\.ai\s+import\s+[^\n]*\bsequence_classifier\b|^\s*import\s+sequence_classifier\b|"
    r"\bSequenceClassifier\b)", re.M)
users = []
for root, _dirs, files in os.walk("app"):
    for fn in files:
        if fn.endswith(".py"):
            path = os.path.join(root, fn).replace("\\", "/")
            if path.endswith("app/ai/sequence_classifier.py"):
                continue
            text = new if path.replace("/", os.sep) == REC.replace("/", os.sep) else open(path, encoding="utf-8").read()
            if USES_LSTM.search(text):
                users.append(path)
if users:
    sys.exit("STOPPED - nothing was changed. These files still use the LSTM module:\n  " + "\n  ".join(users) +
             "\nSend me those lines.")

if n:
    shutil.copy(REC, REC + ".bak")
    with open(REC, "w", encoding="utf-8", newline="") as f:
        f.write(new)
    print(f"recognition.py: LSTM import replaced ({n} line). Backup: {REC}.bak")
else:
    print("recognition.py: no LSTM import found (already clean).")

# 3) archive the LSTM files
os.makedirs(ARCHIVE, exist_ok=True)
for path in MOVE:
    if os.path.exists(path):
        shutil.move(path, os.path.join(ARCHIVE, os.path.basename(path)))
        print("moved to archive:", path)
print("Done. Restart the backend; the terminal should print nothing about the LSTM.")
