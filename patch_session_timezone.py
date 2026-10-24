r"""Makes session.py send UTC times as ISO strings ending in 'Z'.
Run from E:\Projects\hamkalaam_backend:  python patch_session_timezone.py
Makes a backup first (app/routes/session.py.bak)."""
import re, shutil, sys

P = "app/routes/session.py"
src = open(P, encoding="utf-8").read()
if "def _iso(" in src:
    sys.exit("Already patched.")

helper = '''def _iso(dt):
    """UTC datetime -> ISO string with a trailing 'Z', so the app knows it is
    UTC and can show it in the phone's own time zone."""
    return dt.isoformat() + "Z" if dt else None


'''
marker = "def _serialize_sentence(s):"
if marker not in src:
    sys.exit("Could not find _serialize_sentence - paste me the top of the file.")
new = src.replace(marker, helper + marker, 1)
new, n = re.subn(r"(\w+\.\w+)\.isoformat\(\) if \1 else None", r"_iso(\1)", new)
if n == 0:
    sys.exit("No isoformat() lines found to change.")
shutil.copy(P, P + ".bak")
open(P, "w", encoding="utf-8").write(new)
print(f"OK: changed {n} timestamp fields. Backup: {P}.bak")
