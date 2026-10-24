r"""Lowers MIN_HANDS_RATIO in app/ai/sequence_classifier.py from 0.15 to 0.03.
Run from E:\Projects\hamkalaam_backend:  python patch_no_hand_guard.py
Makes a backup (sequence_classifier.py.bak) first."""
import re, shutil, sys
P = "app/ai/sequence_classifier.py"
src = open(P, encoding="utf-8").read()
new, n = re.subn(r"(MIN_HANDS_RATIO\s*=\s*)[0-9.]+", r"\g<1>0.03", src, count=1)
if n == 0:
    sys.exit("MIN_HANDS_RATIO not found - paste me the top 40 lines of the file.")
shutil.copy(P, P + ".bak")
open(P, "w", encoding="utf-8").write(new)
print("OK: MIN_HANDS_RATIO is now 0.03. Backup saved as", P + ".bak")
