r"""Makes the Random Forest the ONLY word model of the backend and archives the LSTM.

Run ONCE from E:\Projects\hamkalaam_backend (venv active):   python use_forest_only.py

1. recognition.py (backup: recognition.py.bak)
     - the import of the LSTM module is replaced by the forest module's ModelUnavailable
     - every  SequenceClassifier(...)  call is replaced by  build_word_classifier()
2. Checks that no other file still uses the LSTM module. If one does, it STOPS (nothing changed).
3. Moves (does not delete) the LSTM files to ..\lstm_archive\ :
     app/ai/sequence_classifier.py, app/ai/model/psl_lstm.keras, app/ai/model/labels.json
   results.json stays (Model Management shows the LSTM-vs-forest comparison from it).
"""
import os
import re
import shutil
import sys

REC = "app/routes/recognition.py"
ARCHIVE = os.path.join("..", "lstm_archive")
MOVE = ["app/ai/sequence_classifier.py", "app/ai/model/psl_lstm.keras", "app/ai/model/labels.json"]

for need in ("app/ai/forest_classifier.py", "app/ai/word_classifier.py",
             "app/ai/model/psl_forest.joblib", "app/ai/model/forest_labels.json"):
    if not os.path.isfile(need):
        sys.exit(f"Missing: {need}\nInstall the Random Forest zips and copy the 3 forest files into "
                 "app/ai/model/ first, then run this again.")

with open(REC, "r", encoding="utf-8", newline="") as f:
    raw = f.read()
text = raw.replace("\r\n", "\n")
crlf = "\r\n" in raw

# ---- 1a) the LSTM import (one line, or a bracketed multi-line import)
imp = re.compile(r"^[ \t]*from[ \t]+[\w\.]*sequence_classifier[ \t]+import[ \t]+(\([^)]*\)|[^\n]*)[ \t]*\n", re.M)
names_ok = {"SequenceClassifier", "ModelUnavailable"}
for m in imp.finditer(text):
    names = {n.strip() for n in re.split(r"[,\s()]+", m.group(1)) if n.strip() and n.strip() != "as"}
    extra = names - names_ok
    if extra:
        sys.exit("STOPPED - recognition.py imports other things from the LSTM module: "
                 f"{sorted(extra)}. Nothing was changed. Send me the top 30 lines of recognition.py.")
n_imp = len(imp.findall(text))
text = imp.sub("", text)

# ---- 1b) every SequenceClassifier(...) call  (arguments without nested brackets)
call = re.compile(r"(?<![\w\.])SequenceClassifier\s*\(([^()]*)\)")
n_call = len(call.findall(text))
text = call.sub("build_word_classifier()", text)

# ---- 1c) make sure the two imports we need are there
need_lines = []
if "from app.ai.forest_classifier import ModelUnavailable" not in text:
    need_lines.append("from app.ai.forest_classifier import ModelUnavailable")
if "build_word_classifier" in text and "from app.ai.word_classifier import build_word_classifier" not in text:
    need_lines.append("from app.ai.word_classifier import build_word_classifier")
if need_lines:
    first_use = re.search(r"^(?:[ \t]*_sequence_classifier\b|@\w+_bp\.route|def |class )", text, re.M)
    at = first_use.start() if first_use else len(text)
    text = text[:at] + "\n".join(need_lines) + "\n" + text[at:]

# ---- still any LSTM name left in recognition.py?
if re.search(r"(?<![\w\.])SequenceClassifier\b", text) or "sequence_classifier import" in text:
    lines = [l for l in text.splitlines() if "SequenceClassifier" in l or "sequence_classifier import" in l]
    sys.exit("STOPPED - recognition.py still mentions the LSTM in a way I do not understand. "
             "Nothing was changed. Send me these lines:\n  " + "\n  ".join(lines))

# ---- 2) any OTHER file using the LSTM module?
USES = re.compile(r"(from\s+[\w\.]*sequence_classifier\s+import|from\s+app\.ai\s+import\s+[^\n]*\bsequence_classifier\b|"
                  r"^\s*import\s+[\w\.]*sequence_classifier\b|(?<![\w\.])SequenceClassifier\b)", re.M)
users = []
for root, _dirs, files in os.walk("app"):
    for fn in files:
        if not fn.endswith(".py"):
            continue
        path = os.path.join(root, fn).replace("\\", "/")
        if path.endswith("app/ai/sequence_classifier.py"):
            continue
        body = text if path == REC else open(path, encoding="utf-8").read()
        if USES.search(body):
            users.append(path)
if users:
    sys.exit("STOPPED - nothing was changed. These files still use the LSTM module:\n  " + "\n  ".join(users) +
             "\nSend me those lines.")

# ---- write recognition.py
if n_imp or n_call or need_lines:
    shutil.copy(REC, REC + ".bak")
    out = text.replace("\n", "\r\n") if crlf else text
    with open(REC, "w", encoding="utf-8", newline="") as f:
        f.write(out)
    print(f"recognition.py updated ({n_imp} LSTM import, {n_call} SequenceClassifier call). Backup: {REC}.bak")
else:
    print("recognition.py: already clean.")

# ---- 3) archive the LSTM files
os.makedirs(ARCHIVE, exist_ok=True)
for path in MOVE:
    if os.path.exists(path):
        shutil.move(path, os.path.join(ARCHIVE, os.path.basename(path)))
        print("moved to archive:", path)
print("Done. Start the backend with:  python run.py")
