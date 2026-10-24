# -*- coding: utf-8 -*-
"""Run:  python test_sentence_engine.py   (no extra packages needed)"""
import unittest
from app.ai.sentence_engine import generate, generate_with_rule

# (words, expression, expected Urdu)
CASES = [
    # greetings
    (["hello"], None, "ہیلو۔"),
    (["assalam-o-alaikum"], None, "السلام علیکم۔"),
    (["thankyou"], None, "شکریہ۔"),
    (["welldone"], None, "شاباش!"),
    (["goodbye"], None, "خدا حافظ۔"),
    (["hello", "thankyou"], None, "ہیلو، شکریہ۔"),
    (["hello", "hello"], None, "ہیلو۔"),
    (["hello", "water", "want"], None, "ہیلو۔ مجھے پانی چاہیے۔"),
    # want / need
    (["water", "want"], None, "مجھے پانی چاہیے۔"),
    (["want", "water"], None, "مجھے پانی چاہیے۔"),
    (["water", "need"], None, "مجھے پانی کی ضرورت ہے۔"),
    (["doctor", "need"], None, "مجھے ڈاکٹر کی ضرورت ہے۔"),
    (["we", "water", "want"], None, "ہمیں پانی چاہیے۔"),
    (["you", "water", "want"], None, "آپ کو پانی چاہیے۔"),
    (["water", "tissue", "want"], None, "مجھے پانی اور ٹشو چاہییں۔"),
    (["color_pencils", "want"], None, "مجھے رنگین پنسلیں چاہییں۔"),
    (["water", "also", "want"], None, "مجھے پانی بھی چاہیے۔"),
    (["water", "want"], "Question", "کیا آپ کو پانی چاہیے؟"),
    (["water", "need"], "Question", "کیا آپ کو پانی کی ضرورت ہے؟"),
    (["water", "want"], "Negation", "مجھے پانی نہیں چاہیے۔"),
    (["water", "need"], "Negation", "مجھے پانی کی ضرورت نہیں ہے۔"),
    (["need", "nothing"], None, "مجھے کچھ نہیں چاہیے۔"),
    (["want"], None, "مجھے کچھ چاہیے۔"),
    (["all", "water", "want"], None, "مجھے سب پانی چاہیے۔"),
    # help
    (["help"], None, "مدد!"),
    (["help", "need"], None, "مجھے مدد چاہیے۔"),
    (["doctor", "help"], None, "مجھے ڈاکٹر کی مدد چاہیے۔"),
    (["we", "help", "need"], None, "ہمیں مدد چاہیے۔"),
    (["help"], "Question", "کیا آپ کو مدد چاہیے؟"),
    # questions
    (["doctor", "where"], None, "ڈاکٹر کہاں ہے؟"),
    (["where", "atm"], None, "اے ٹی ایم کہاں ہے؟"),
    (["you", "where"], None, "آپ کہاں ہیں؟"),
    (["we", "where"], None, "ہم کہاں ہیں؟"),
    (["where"], None, "کہاں؟"),
    (["doctor", "bed", "where"], None, "ڈاکٹر اور بستر کہاں ہیں؟"),
    (["you", "who"], None, "آپ کون ہیں؟"),
    (["doctor", "who"], None, "ڈاکٹر کون ہے؟"),
    (["who"], None, "کون ہے؟"),
    # hungry
    (["hungry"], None, "مجھے بھوک لگی ہے۔"),
    (["we", "hungry"], None, "ہمیں بھوک لگی ہے۔"),
    (["you", "hungry"], None, "کیا آپ کو بھوک لگی ہے؟"),
    (["hungry"], "Question", "کیا آپ کو بھوک لگی ہے؟"),
    (["hungry"], "Negation", "مجھے بھوک نہیں لگی۔"),
    # give / bring / come
    (["water", "give"], None, "مجھے پانی دیجیے۔"),
    (["give"], None, "دیجیے۔"),
    (["tissue", "bring"], None, "ٹشو لائیے۔"),
    (["bring"], None, "لائیے۔"),
    (["come"], None, "آئیے۔"),
    (["you", "come"], None, "آپ آئیے۔"),
    (["we", "come"], None, "ہم آ رہے ہیں۔"),
    (["doctor", "come"], None, "ڈاکٹر کو بلائیے۔"),
    # complaint / donttouch / nothing
    (["ihaveacomplaint"], None, "مجھے شکایت ہے۔"),
    (["ihaveacomplaint", "doctor"], None, "مجھے ڈاکٹر کے بارے میں شکایت ہے۔"),
    (["donttouch"], None, "ہاتھ مت لگائیں۔"),
    (["water", "donttouch"], None, "پانی کو ہاتھ مت لگائیں۔"),
    (["nothing"], None, "کچھ نہیں۔"),
    # mine
    (["water", "mine"], None, "یہ پانی میرا ہے۔"),
    (["umbrella", "mine"], None, "یہ چھتری میری ہے۔"),
    (["color_pencils", "mine"], None, "یہ رنگین پنسلیں میری ہیں۔"),
    (["mine"], None, "یہ میرا ہے۔"),
    # arrival
    (["airplane", "arrival"], None, "ہوائی جہاز کی آمد۔"),
    (["arrival"], None, "آمد۔"),
    # subject + noun, fallback
    (["you", "doctor"], None, "آپ ڈاکٹر ہیں۔"),
    (["you", "doctor"], "Question", "کیا آپ ڈاکٹر ہیں؟"),
    (["dog"], None, "کتا۔"),
    (["dog"], "Question", "کتا؟"),
    (["dog", "cow"], None, "کتا اور گائے۔"),
    (["absolutely"], None, "بالکل۔"),
]


class SentenceEngineTests(unittest.TestCase):
    def test_table(self):
        failures = []
        for words, expr, expected in CASES:
            got = generate(words, expr)
            if got != expected:
                failures.append(f"{words} [{expr}] -> {got!r}  (expected {expected!r})")
        self.assertEqual(failures, [], "\n" + "\n".join(failures))

    def test_empty_input(self):
        self.assertEqual(generate([]), "")
        self.assertEqual(generate(None), "")

    def test_unknown_label_never_crashes(self):
        self.assertTrue(generate(["not_a_real_word"]).endswith("۔"))

    def test_never_empty_for_real_words(self):
        import json, os
        path = os.path.join("app", "ai", "psl_words_ur.json")
        with open(path, encoding="utf-8") as f:
            labels = [w["label"] for w in json.load(f)]
        for l in labels:
            for expr in (None, "Question", "Negation"):
                self.assertTrue(generate([l], expr), f"empty for {l} [{expr}]")

    def test_rule_name_reported(self):
        self.assertEqual(generate_with_rule(["doctor", "where"])[1], "where_noun")


if __name__ == "__main__":
    print(f"{len(CASES)} table cases")
    unittest.main(verbosity=1)
