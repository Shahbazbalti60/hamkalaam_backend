# -*- coding: utf-8 -*-
"""Rule-based Urdu sentence engine for HamKalaam.

Turns an ordered list of recognised PSL word labels (plus the facial
expression context) into a short, grammatical Urdu sentence.

    generate(["water", "want"])                 -> "مجھے پانی چاہیے۔"
    generate(["doctor", "where"])               -> "ڈاکٹر کہاں ہے؟"
    generate(["water", "want"], "Question")     -> "کیا آپ کو پانی چاہیے؟"

How it works: it does NOT depend on the signing order. It sorts the
words into groups (pronoun, question word, verb, noun, ...) and picks
the first pattern that fits. If no pattern fits it falls back to
joining the Urdu words, so it never fails or returns an empty string.

Expression context used: "Question" and "Negation". The other labels
(Affirmation, Neutral, Happy, Sad) do not change the sentence yet.

All Urdu text is a DRAFT: please have a native speaker review it.
"""
import json
import os
from typing import List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_HERE, "psl_words_ur.json"), encoding="utf-8") as _f:
    URDU = {w["label"]: w["urdu"] for w in json.load(_f)}

GREETINGS = {"assalam-o-alaikum", "hello", "goodbye", "goodmorning",
             "have_a_good_day", "thankyou", "excuseme", "welldone"}
PRONOUNS = {"you", "we"}
QWORDS = {"where", "who"}
NEED_WANT = {"need", "want"}
MODIFIERS = {"all", "both"}          # go in front of the next noun
SPECIAL = {"mine", "nothing", "donttouch", "ihaveacomplaint", "arrival",
           "help", "hungry", "give", "bring", "come", "also", "absolutely"}

# gender of nouns (f = feminine, anything else = masculine); plural flag
FEMININE = {"bicycle", "cupboard", "umbrella", "cow", "beak", "beard",
            "policecar", "lifejacket", "color_pencils"}
PLURAL = {"color_pencils"}

DATIVE = {"you": "آپ کو", "we": "ہمیں", None: "مجھے"}


def _u(label: str) -> str:
    return URDU.get(label, label)


def _join_and(items: List[str]) -> str:
    return " اور ".join(items)


def _end(text: str, question: bool) -> str:
    return text + ("؟" if question else "۔")


def _noun_phrases(words: List[str]) -> List[Tuple[str, str]]:
    """Nouns in signing order as (label, text), with all/both attached."""
    out, prefix = [], ""
    for w in words:
        if w in MODIFIERS:
            prefix = _u(w) + " "
        elif (w not in GREETINGS and w not in PRONOUNS and w not in QWORDS
              and w not in NEED_WANT and w not in SPECIAL):
            out.append((w, prefix + _u(w)))
            prefix = ""
    if prefix:  # "all" / "both" with no noun after it
        out.append(("", prefix.strip()))
    return out


def _clause(words: List[str], question: bool, negation: bool) -> Tuple[str, str]:
    """Build one sentence from the non-greeting words. Returns (text, rule)."""
    has = set(words)
    subj = "you" if "you" in has else ("we" if "we" in has else None)
    nps = _noun_phrases(words)
    labels = [l for l, _ in nps if l]
    texts = [t for _, t in nps]
    plural = any(l in PLURAL for l in labels) or len(texts) > 1
    obj = _join_and(texts)
    also = " بھی" if "also" in has else ""

    # --- question words ------------------------------------------------
    if "where" in has:
        if subj:
            return _end(f"{_u(subj)} کہاں ہیں", True), "where_subject"
        if texts:
            verb = "ہیں" if plural else "ہے"
            return _end(f"{obj} کہاں {verb}", True), "where_noun"
        return "کہاں؟", "where_alone"
    if "who" in has:
        if subj:
            return _end(f"{_u(subj)} کون ہیں", True), "who_subject"
        if texts:
            verb = "ہیں" if plural else "ہے"
            return _end(f"{obj} کون {verb}", True), "who_noun"
        return "کون ہے؟", "who_alone"

    # --- complaint / don't touch ----------------------------------------
    if "ihaveacomplaint" in has:
        if texts:
            return _end(f"مجھے {obj} کے بارے میں شکایت ہے", False), "complaint_about"
        return "مجھے شکایت ہے۔", "complaint"
    if "donttouch" in has:
        if texts:
            return _end(f"{obj} کو ہاتھ مت لگائیں", False), "donttouch_noun"
        return "ہاتھ مت لگائیں۔", "donttouch"

    # --- hungry ----------------------------------------------------------
    if "hungry" in has:
        if question or subj == "you":
            return "کیا آپ کو بھوک لگی ہے؟", "hungry_question"
        who = "ہمیں" if subj == "we" else "مجھے"
        if negation:
            return f"{who} بھوک نہیں لگی۔", "hungry_negation"
        return f"{who} بھوک لگی ہے۔", "hungry"

    # --- need / want / help ----------------------------------------------
    if has & NEED_WANT or "help" in has:
        dative = DATIVE[subj]
        if question and subj is None:
            dative = "آپ کو"
        lead = "کیا " if question else ""
        end_q = question

        if "nothing" in has:                      # "I need nothing"
            return _end(f"{lead}{dative} کچھ نہیں چاہیے", end_q), "need_nothing"

        if "help" in has:
            thing = f"{obj} کی مدد" if texts else "مدد"
            if not (has & NEED_WANT) and not texts and not subj and not question:
                return "مدد!", "help_alone"
            return _end(f"{lead}{dative} {thing} چاہیے", end_q), "help"

        thing = obj if texts else "کچھ"
        if not texts and not question:
            thing = "کچھ"
        thing += also
        if "want" in has or ("need" in has and not texts):
            verb = "چاہییں" if plural else "چاہیے"
            neg = " نہیں" if negation else ""
            return _end(f"{lead}{dative} {thing}{neg} {verb}", end_q), "want"
        neg = "نہیں ہے" if negation else "ہے"
        return _end(f"{lead}{dative} {thing} کی ضرورت {neg}", end_q), "need"

    # --- give / bring / come ----------------------------------------------
    if "give" in has:
        if texts:
            return _end(f"مجھے {obj}{also} دیجیے", False), "give"
        return "دیجیے۔", "give_alone"
    if "bring" in has:
        if texts:
            return _end(f"{obj}{also} لائیے", False), "bring"
        return "لائیے۔", "bring_alone"
    if "come" in has:
        if subj == "we":
            return "ہم آ رہے ہیں۔", "come_we"
        if subj == "you":
            return "آپ آئیے۔", "come_you"
        if texts:
            return _end(f"{obj} کو بلائیے", False), "come_noun"
        return "آئیے۔", "come_alone"

    # --- mine --------------------------------------------------------------
    if "mine" in has:
        if not labels:
            return _end("یہ میرا ہے", question), "mine_alone"
        last = labels[-1]
        if plural:
            form = "میری ہیں" if last in FEMININE else "میرے ہیں"
        else:
            form = "میری ہے" if last in FEMININE else "میرا ہے"
        return _end(f"یہ {obj} {form}", question), "mine"

    # --- arrival / nothing -----------------------------------------------
    if "arrival" in has:
        if texts:
            return _end(f"{obj} کی آمد", question), "arrival_noun"
        return _end("آمد", question), "arrival"
    if "nothing" in has:
        if texts:
            return _end(f"{obj} کچھ نہیں", question), "nothing_noun"
        return _end("کچھ نہیں", question), "nothing"

    # --- subject + noun ("you doctor") ---------------------------------------
    if subj and texts:
        text = f"{_u(subj)} {obj} ہیں"
        return _end(("کیا " if question else "") + text, question), "subject_noun"

    # --- fallback: just the words ----------------------------------------------
    parts = ([_u(subj)] if subj else []) + texts
    if "absolutely" in has:
        parts.append(_u("absolutely"))
    if not parts:
        if "also" in has:
            return _end(_u("also"), question), "also_alone"
        return "", "empty"
    sep = " اور " if len(texts) > 1 and not subj else " "
    return _end(sep.join(parts) + also, question), "fallback"


def generate_with_rule(word_sequence: List[str],
                       expression_context: Optional[str] = None) -> Tuple[str, str]:
    """Returns (urdu_sentence, rule_name). rule_name is handy for debugging."""
    words = [w for w in (word_sequence or []) if w]
    if not words:
        return "", "empty"
    question = expression_context == "Question"
    negation = expression_context == "Negation"

    greetings = []
    for w in words:
        if w in GREETINGS and w not in greetings:
            greetings.append(w)
    rest = [w for w in words if w not in GREETINGS]

    greet_text = ""
    if greetings:
        greet_text = "، ".join(_u(g) for g in greetings)
        greet_text += "!" if greetings[-1] == "welldone" else "۔"

    if not rest:
        return greet_text, "greeting"

    text, rule = _clause(rest, question, negation)
    if greet_text:
        return f"{greet_text} {text}", "greeting+" + rule
    return text, rule


def generate(word_sequence: List[str],
             expression_context: Optional[str] = None) -> str:
    """Urdu sentence for the given word labels. Never raises."""
    try:
        return generate_with_rule(word_sequence, expression_context)[0]
    except Exception:
        return " ".join(_u(w) for w in (word_sequence or [])) + "۔"
