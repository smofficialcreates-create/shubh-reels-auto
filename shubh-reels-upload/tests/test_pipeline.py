import datetime as dt
import unittest
from unittest import mock

from reelbot import grounding, research, script_gen, topic
from reelbot.config import Config

EXCERPT = (
    "Researchers at a design lab tracked how people work with AI assistants for a month. "
    "They found that people who wrote one clear goal before opening the assistant finished their tasks sooner. "
    "Workers who kept many tabs open lost about forty minutes a day to switching between them. "
    "The team recommends pasting messy notes into an assistant and asking it to rank the tasks. "
    "Closing everything except the top task helped people stay focused for longer stretches."
)

RAW = {
    "topic": "One clear goal before you open the assistant", "category": "productivity_methods", "tag": "FOCUS",
    "hook_options": ["You lose forty minutes a day without noticing.", "Your tabs are quietly stealing your afternoon.",
                     "Stop opening the assistant before this one step."],
    "hook_mechanisms": ["specific outcome", "problem", "pattern interrupt"], "hook_focus": "forty",
    "sentences": [
        "A design lab tracked how people use assistants.",
        "Workers with many tabs lost about forty minutes a day.",
        "First, write one clear goal before you start.",
        "Second, paste your messy notes and ask it to rank the tasks.",
        "Third, close everything except the top task.",
        "That is how the team kept people focused longer."],
    "key_points": [{"sentence": 3, "label": "Write one goal"}, {"sentence": 4, "label": "Rank the tasks"},
                   {"sentence": 5, "label": "Close the rest"}],
    "takeaway": "Start with one goal, then let an assistant rank your notes.", "demo": None,
    "emphasis": ["lab", "forty", "goal", "notes", "close", "focused", "save"],
    "evidence": [{"sentence": 2, "quote": "lost about forty minutes a day to switching between them"},
                 {"sentence": 4, "quote": "pasting messy notes into an assistant and asking it to rank the tasks"},
                 {"sentence": 5, "quote": "Closing everything except the top task helped people stay focused"}],
    "opinions": [], "cta_type": "save", "cta": "Save this for your next work session.",
    "cover_text": "Stop losing forty minutes a day", "caption": "You lose forty minutes a day.\nOne goal first.\nfocus, tabs\nAI-generated video.",
    "hashtags": ["#productivity", "#aitools", "#focus"],
}
GOOD = {"hook": 17, "accuracy": 18, "value": 16, "retention": 16, "visual": 8, "originality": 7, "unsupported_claims": [], "problems": []}


def plan():
    c = research.Candidate(1, "Design lab studies AI focus habits", "https://example.com/a", "Example News", "2026-10-28", 2.0)
    return {"mode": "source", "cand": c, "category": "productivity_methods", "angle": "x", "scores": {}, "total": 80, "excerpt": EXCERPT}


def run_attempt(raw, review):
    calls = iter([raw, review])
    with mock.patch("reelbot.llm.chat_json", side_effect=lambda *a, **k: next(calls)):
        return script_gen.attempt(Config(), plan(), [], [], dt.date(2026, 10, 30))


class Rubric(unittest.TestCase):
    def test_weighted_total_is_computed_in_code(self):
        self.assertEqual(topic.total_score({k: 10 for k in topic.WEIGHTS}), 100.0)
        self.assertEqual(topic.total_score({k: 5 for k in topic.WEIGHTS}), 50.0)

    def test_category_rotation(self):
        h = [{"category": "a"}, {"category": "b"}]
        self.assertLess(topic.category_bonus("b", h), 0)
        self.assertGreater(topic.category_bonus("c", h), 0)


class Grounding(unittest.TestCase):
    def test_number_words(self):
        self.assertEqual(sorted(grounding.numbers_in("forty minutes, twenty five, 1,200, two hundred and fifty")),
                         [25.0, 40.0, 250.0, 1200.0])

    def test_wrong_quote_and_invented_number_fail(self):
        s = dict(RAW, sentences=RAW["sentences"][:1] + ["Workers lost ninety minutes a day."] + RAW["sentences"][2:],
                 hook="x", evidence=[{"sentence": 2, "quote": "this sentence is not in the article at all"}])
        _ok, issues = grounding.verify(s, EXCERPT)
        self.assertTrue(any("not in the source" in i for i in issues))
        self.assertTrue(any("verbatim quote" in i for i in issues))


class Attempt(unittest.TestCase):
    def test_good_script_passes_and_is_scored(self):
        script, issues = run_attempt(RAW, GOOD)
        self.assertEqual(issues, [])
        self.assertEqual(script["quality"]["total"], 82)
        self.assertEqual(len(script["evidence_verified"]), 3)

    def test_invented_quote_is_rejected_before_review(self):
        bad = dict(RAW, evidence=[{"sentence": 2, "quote": "an invented quotation that never appeared"}] * 3)
        script, issues = run_attempt(bad, GOOD)
        self.assertIsNone(script)
        self.assertTrue(any("verbatim quote" in i for i in issues))

    def test_low_accuracy_or_unsupported_claims_rejected(self):
        script, issues = run_attempt(RAW, dict(GOOD, accuracy=12))
        self.assertIsNone(script)
        script, issues = run_attempt(RAW, dict(GOOD, unsupported_claims=["tool is free"]))
        self.assertIsNone(script)

    def test_digits_and_free_claims_are_blocked(self):
        bad = dict(RAW, sentences=RAW["sentences"][:1] + ["Workers lost 40 minutes, and the app is free."] + RAW["sentences"][2:])
        script, issues = run_attempt(bad, GOOD)
        self.assertIsNone(script)
        self.assertTrue(any("digits" in i for i in issues) and any("pricing" in i for i in issues))


class Research(unittest.TestCase):
    def test_gather_filters_stale_and_blocked_hosts(self):
        now = dt.datetime(2026, 10, 30, tzinfo=dt.timezone.utc)
        rss = ("<rss><channel>"
               "<item><title>Fresh useful story about AI notes</title><link>https://a.com/1</link><pubDate>Wed, 28 Oct 2026 10:00:00 +0000</pubDate></item>"
               "<item><title>Old story from long ago about tabs</title><link>https://a.com/2</link><pubDate>Mon, 01 Jun 2026 10:00:00 +0000</pubDate></item>"
               "<item><title>A video about something interesting</title><link>https://youtube.com/watch?v=1</link><pubDate>Wed, 28 Oct 2026 10:00:00 +0000</pubDate></item>"
               "</channel></rss>")
        resp = mock.Mock(text=rss)
        src = {"feeds": [["Ex", "https://feed.example/rss"]], "hn": [], "skip_hosts": ["youtube.com"], "max_age_days": 10}
        with mock.patch("reelbot.research._get", return_value=resp):
            got = research.gather(now, src)
        self.assertEqual([c.url for c in got], ["https://a.com/1"])

    def test_unreachable_sources_give_empty_list(self):
        with mock.patch("reelbot.research._get", return_value=None):
            self.assertEqual(research.gather(None, {"feeds": [["x", "https://x.test"]], "hn": ["https://y.test"]}), [])


if __name__ == "__main__":
    unittest.main()
