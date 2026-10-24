r"""Seed the 71 PSL words (with Urdu + domain) into DatasetWord.
Run from E:\Projects\hamkalaam_backend with the venv active:  python seed_words.py
Safe to run many times: it updates existing words and adds missing ones."""
import json, os
from app import create_app, db
from app.models.user import DatasetWord

PATH = os.path.join(os.path.dirname(__file__), "app", "ai", "psl_words_ur.json")

app = create_app()
with app.app_context():
    words = json.load(open(PATH, encoding="utf-8"))
    added = updated = 0
    for w in words:
        row = DatasetWord.query.filter_by(english_gloss=w["label"]).first()
        if row is None:
            row = DatasetWord(urdu_word=w["urdu"], english_gloss=w["label"], domain=w["domain"])
            db.session.add(row)
            added += 1
        else:
            row.urdu_word = w["urdu"]
            row.domain = w["domain"]
            updated += 1
    db.session.commit()
    print(f"Done. added={added} updated={updated} total_in_db={DatasetWord.query.count()}")
