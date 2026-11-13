import datetime as dt
import json
import unittest
from unittest import mock

from reelbot import grounding, plan as planner, research, script_gen, topic
from reelbot.config import Config
from reelbot.sample import SAMPLE

EXCERPT = (
    "Researchers at a design lab tracked how people work with AI assistants for a month. "
    "They found that people who wrote one clear goal before opening the assistant finished their tasks sooner. "
    "Workers who kept many tabs open lost about forty minutes a day to switching between them. "
    "The team recommends pasting messy notes into an assistant and asking it to rank the tasks. "
    "Closing everything except the top task helped people stay focused for longer stretches."
)

# a source-style script used only to test the grounding checks
RAW = {
    "sentences": [
        "A design lab tracked how people use assistants.",
        "Workers with many tabs lost about forty minutes a day.",
        "First, write one clear goal before you start.",
        "Second, paste your messy notes and ask it to rank the tasks.",
        "Third, close everything except the top task.",
        "That is how the team kept people focused longer."],
    "evidence": [{"sentence": 2, "quote": "lost about forty minutes a day to switching between them"},
                 {"sentence": 4, "quote": "pasting messy notes into an assistant and asking it to rank the tasks"},
                 {"sentence": 5, "quote": "Closing everything except the top task helped people stay focused"}],
}

# a full explainer in the shape the writer returns, built from the built-in sample
DEEP = dict(SAMPLE)
DEEP.update(hook_options=[SAMPLE["hook"], "Your AI is guessing, not searching.", "What an AI does before it answers you."],
            hook_mechanisms=["pattern interrupt", "contrarian", "curiosity gap"], evidence=[], opinions=[])
GOOD = {"hook": 13, "accuracy": 18, "depth": 16, "clarity": 13, "retention": 12, "visuals": 8, "originality": 4,
        "unsupported_claims": [], "problems": []}


def explainer_plan():
    e = {"title": "How an AI assistant actually answers you", "category": "practical_ai_tutorials", "analogy": "autocomplete",
         "stages": ["tokens", "desk", "prediction"], "catch": "it can be wrong"}
    return {"mode": "explainer", "entry": e, "pillar": e["category"], "seed": e["title"], "category": e["category"]}


def source_plan():
    c = research.Candidate(1, "Design lab studies AI focus habits", "https://example.com/a", "Example News", "2026-10-28", 2.0)
    return {"mode": "source", "cand": c, "category": "productivity_methods", "angle": "x", "scores": {}, "total": 80, "excerpt": EXCERPT}


def run_attempt(raw, review, plan=None):
    calls = iter([raw, review])
    with mock.patch("reelbot.llm.chat_json", side_effect=lambda *a, **k: next(calls)):
        return script_gen.attempt(Config(), plan or explainer_plan(), [], [], dt.date(2026, 10, 30))


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
    def test_good_deep_script_passes_and_is_scored(self):
        script, issues = run_attempt(DEEP, GOOD)
        self.assertEqual(issues, [])
        self.assertEqual(script["quality"]["total"], 84)
        self.assertGreaterEqual(len(script["scenes"]), 6)

    def test_thin_script_is_rejected(self):
        thin = dict(DEEP, sentences=DEEP["sentences"][:6], scenes=[])
        script, issues = run_attempt(thin, GOOD)
        self.assertIsNone(script)
        self.assertTrue(any("too thin" in i for i in issues))

    def test_invented_quote_is_rejected_in_source_mode(self):
        bad = dict(DEEP, evidence=[{"sentence": 2, "quote": "an invented quotation that never appeared anywhere"}] * 3)
        script, issues = run_attempt(bad, GOOD, source_plan())
        self.assertIsNone(script)
        self.assertTrue(any("verbatim quote" in i for i in issues))

    def test_low_depth_accuracy_or_unsupported_claims_rejected(self):
        for review in (dict(GOOD, accuracy=12), dict(GOOD, depth=9), dict(GOOD, unsupported_claims=["tool is free"])):
            script, issues = run_attempt(DEEP, review)
            self.assertIsNone(script)

    def test_digits_and_free_claims_are_blocked(self):
        s = list(DEEP["sentences"])
        s[1] = "Workers lost 40 minutes, and the app is free."
        script, issues = run_attempt(dict(DEEP, sentences=s), GOOD)
        self.assertIsNone(script)
        self.assertTrue(any("digits" in i for i in issues) and any("pricing" in i for i in issues))


class Scenes(unittest.TestCase):
    def lex(self):
        from vendor.ig_skill import humanize
        return humanize.load_lexicon()

    def test_counter_needs_a_spoken_number(self):
        body = ["Workers lost forty minutes a day.", "Everyone else lost time too."]
        ok = script_gen.finalize_scenes([{"from": 1, "to": 1, "visual": {"type": "counter", "value": 40, "unit": "min", "label": "lost per day"}}], body, self.lex())
        no = script_gen.finalize_scenes([{"from": 2, "to": 2, "visual": {"type": "counter", "value": 90, "unit": "min", "label": "invented"}}], body, self.lex())
        self.assertEqual(len(ok), 1)
        self.assertEqual(no, [])

    def test_visual_text_cannot_carry_prices_or_digits(self):
        body = ["One plan beats ten.", "Pick it."]
        got = script_gen.finalize_scenes([{"from": 1, "to": 2, "visual": {"type": "definition", "term": "Plan", "meaning": "It costs 5 dollars"}}], body, self.lex())
        self.assertEqual(got, [])

    def test_repeated_visual_types_are_flagged(self):
        sc = [{"from": i, "to": i, "visual": {"type": t}} for i, t in enumerate(["flow", "flow", "compare", "keyword"], 1)]
        self.assertTrue(any("same visual type" in x for x in script_gen.scene_issues({"scenes": sc})))

    def test_gaps_are_closed(self):
        sc = [{"from": 2, "to": 3, "visual": {"type": "keyword"}}, {"from": 6, "to": 7, "visual": {"type": "flow"}}]
        fixed = planner.repair_scenes(sc, 8)
        self.assertEqual([(x["from"], x["to"]) for x in fixed], [(1, 5), (6, 8)])

    def test_sections_follow_the_scenes_and_never_repeat_a_theme(self):
        class Nar:
            sentence_times = [(i * 4.0, i * 4.0 + 3.5) for i in range(len(SAMPLE["sentences"]) + 4)]
        secs = planner.build_sections(SAMPLE, Nar())
        self.assertEqual([secs[0].kind, secs[1].kind], ["hook", "intro"])      # hook first, then Shubh
        self.assertEqual(secs[0].i0, 0)
        self.assertEqual(secs[-1].kind, "cta")
        self.assertEqual(len(secs), len(SAMPLE["scenes"]) + 3)
        for a, b in zip(secs, secs[1:]):
            self.assertNotEqual(a.theme, b.theme)
            self.assertNotEqual(a.transition, b.transition)
        flow = [x for x in secs if x.prop == "flow"][0]
        self.assertEqual(len(flow.data["abs_starts"]), 3)


class Fallback(unittest.TestCase):
    def test_near_miss_used_only_when_everything_else_failed(self):
        script_gen._NEAR.clear()
        low = dict(GOOD, hook=11, accuracy=17, depth=13, clarity=11, retention=11, visuals=6, originality=3)   # 72
        seq = iter([DEEP, low] * 9)
        with mock.patch("reelbot.script_gen.build_plans", return_value=[explainer_plan()]), \
             mock.patch("reelbot.llm.chat_json", side_effect=lambda *a, **k: next(seq)):
            out = script_gen.generate(Config(reel_format="long"), [], dt.date(2026, 10, 30))
        self.assertEqual(out["mode"], "explainer")
        self.assertEqual(out["quality"]["total"], 72)

    def test_below_fallback_still_fails(self):
        script_gen._NEAR.clear()
        bad = dict(GOOD, hook=8, accuracy=18, depth=10, clarity=8, retention=6, visuals=3, originality=2)    # 55
        seq = iter([DEEP, bad] * 9)
        with mock.patch("reelbot.script_gen.build_plans", return_value=[explainer_plan()]), \
             mock.patch("reelbot.llm.chat_json", side_effect=lambda *a, **k: next(seq)):
            out = script_gen.generate(Config(reel_format="long"), [], dt.date(2026, 10, 30))
        self.assertEqual(out["mode"], "backup")   # a 55/100 live script is never used


class Backup(unittest.TestCase):
    def test_every_backup_lesson_passes_safety(self):
        import json
        bank = json.load(open("data/backup_scripts.json", encoding="utf-8"))
        self.assertGreaterEqual(len(bank), 15)
        from reelbot import safety
        for b in bank:
            self.assertEqual(safety.check_script(b), [], b["topic"])
            self.assertEqual(len(b["emphasis"]), len(b["sentences"]) + 2)

    def test_total_ai_outage_still_gives_a_script_and_rotates(self):
        script_gen._NEAR.clear()
        with mock.patch("reelbot.script_gen.build_plans", side_effect=RuntimeError("boom")):
            a = script_gen.generate(Config(reel_format="long"), [], dt.date(2026, 11, 3))
            b = script_gen.generate(Config(reel_format="long"), [{"topic": a["topic"]}], dt.date(2026, 11, 4))
        self.assertEqual(a["mode"], "backup")
        self.assertNotEqual(a["topic"], b["topic"])

    def test_all_attempts_fail_falls_to_backup(self):
        script_gen._NEAR.clear()
        with mock.patch("reelbot.script_gen.build_plans", return_value=[explainer_plan()]), \
             mock.patch("reelbot.llm.chat_json", side_effect=RuntimeError("down")):
            self.assertEqual(script_gen.generate(Config(reel_format="long"), [], dt.date(2026, 11, 3))["mode"], "backup")


class Order(unittest.TestCase):
    def test_hook_is_spoken_before_the_intro(self):
        from reelbot import brand
        lines = brand.spoken(SAMPLE)
        self.assertEqual(lines[0], SAMPLE["hook"])
        self.assertEqual(lines[1], brand.INTRO)
        self.assertEqual(lines[-1], brand.OUTRO)
        self.assertEqual(len(lines), len(SAMPLE["sentences"]) + 4)

    def test_post_time_wait(self):
        import datetime as d
        from reelbot import publish
        ist = publish.IST
        slept = []
        with mock.patch("reelbot.publish.dt") as m, mock.patch("reelbot.publish.time.sleep", side_effect=slept.append):
            m.timedelta = d.timedelta
            m.datetime.now.return_value = d.datetime(2026, 11, 4, 19, 0, tzinfo=ist)
            publish.wait_until("19:30")                      # 27 minutes to go once the 3 minute lead is taken off
            self.assertAlmostEqual(slept[0], 27 * 60, delta=1)
            slept.clear()
            m.datetime.now.return_value = d.datetime(2026, 11, 4, 14, 0, tzinfo=ist)
            publish.wait_until("19:30")                      # more than 90 minutes away: post now
            self.assertEqual(slept, [])
            m.datetime.now.return_value = d.datetime(2026, 11, 4, 19, 40, tzinfo=ist)
            publish.wait_until("19:30")                      # already past: post now
            self.assertEqual(slept, [])


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


SHORT = {
    "topic": "Ask AI for the question behind your question", "category": "practical_ai_tutorials", "tag": "AI TIP",
    "hook_options": ["You are asking your assistant the wrong question.", "Your first prompt is rarely the real one.",
                     "Why does your assistant give vague answers?"],
    "hook_mechanisms": ["wrong way", "hidden truth", "challenge"], "hook_focus": "wrong",
    "sentences": [
        "Most people type the first question that comes to mind.",
        "That question is usually a rough guess at what they really need.",
        "So the assistant gives a rough guess back.",
        "Try this instead: ask it what question you should be asking.",
        "It will list a few sharper questions and why each one matters.",
        "Pick the best one and ask that, and the answer gets specific."],
    "scenes": [
        {"from": 1, "to": 3, "label": "the problem", "visual": {"type": "compare",
            "left": {"title": "weak way", "rows": ["first thought", "vague answer"]},
            "right": {"title": "better way", "rows": ["sharper question", "specific answer"]}}},
        {"from": 4, "to": 5, "label": "the demo", "visual": {"type": "chat", "prompt": "What question should I really be asking about my career switch?",
            "result": "Ask which skills transfer, which gaps matter most, and who hires for both."}},
        {"from": 6, "to": 6, "label": "the rule", "visual": {"type": "keyword", "word": "specific"}}],
    "takeaway": "Ask for the better question first.", "emphasis": ["question", "guess", "guess", "ask", "sharper", "specific", "tomorrow"],
    "evidence": [], "opinions": [], "cta_type": "question",
    "cta": "What would you ask your assistant first tomorrow?",
    "first_comment": "Which question would you sharpen first? Tell me below.",
    "cover_text": "Ask the better question first", "caption": "You are asking the wrong question.\nAsk for the better question first.\nai prompts, productivity\nAI-generated video.",
    "hashtags": ["#aitools", "#productivity", "#prompts"]}
SHORT_REVIEW = dict(GOOD, depth=12)


def short_plan():
    return dict(explainer_plan(), fmt="short")


class ShortMode(unittest.TestCase):
    def test_short_script_passes_and_is_marked_short(self):
        script, issues = run_attempt(SHORT, SHORT_REVIEW, short_plan())
        self.assertEqual(issues, [])
        self.assertEqual(script["format"], "short")
        self.assertTrue(script["first_comment"].endswith("below."))

    def test_short_needs_chat_demo_and_closing_question(self):
        no_chat = dict(SHORT, scenes=[s for s in SHORT["scenes"] if s["visual"]["type"] != "chat"] + [
            {"from": 4, "to": 5, "label": "x", "visual": {"type": "definition", "term": "Sharper", "meaning": "a question with a clear goal"}}])
        _s, issues = run_attempt(no_chat, SHORT_REVIEW, short_plan())
        self.assertTrue(any("chat" in i for i in issues))
        _s, issues = run_attempt(dict(SHORT, cta="Save this for later."), SHORT_REVIEW, short_plan())
        self.assertTrue(any("question" in i for i in issues))

    def test_deep_script_is_rejected_in_short_mode(self):
        _s, issues = run_attempt(DEEP, GOOD, short_plan())
        self.assertTrue(any("short reel needs" in i for i in issues))

    def test_name_comes_last_in_short_reels(self):
        from reelbot import brand
        script, _ = run_attempt(SHORT, SHORT_REVIEW, short_plan())
        lines = brand.spoken(script)
        self.assertEqual(lines[0], script["hook"])
        self.assertEqual(lines[1], script["sentences"][0])        # payoff straight after the hook
        self.assertEqual(lines[-1], brand.OUTRO_SHORT)
        self.assertNotIn(brand.INTRO, lines)
        self.assertEqual(brand.offset(script), 0)

    def test_sections_cover_a_short_reel(self):
        from reelbot import brand, tts
        script, _ = run_attempt(SHORT, SHORT_REVIEW, short_plan())
        nar = tts.synth_fake(brand.spoken(script))
        secs = planner.build_sections(script, nar)
        self.assertEqual(secs[0].kind, "hook")
        self.assertNotIn("intro", [s.kind for s in secs])
        self.assertEqual(secs[-1].kind, "cta")
        self.assertEqual(secs[-1].i1, len(script["sentences"]) + 2)

    def test_format_selection_and_backup(self):
        d = dt.date(2026, 11, 14)
        self.assertEqual(script_gen.pick_format(Config(reel_format="long"), d), "long")
        self.assertEqual(script_gen.pick_format(Config(reel_format="short"), d), "short")
        self.assertEqual({script_gen.pick_format(Config(reel_format="alternate"), d),
                          script_gen.pick_format(Config(reel_format="alternate"), d + dt.timedelta(days=1))}, {"short", "long"})
        b = script_gen.backup_script([], d, "short")
        self.assertEqual(b["format"], "short")
        self.assertTrue(b["first_comment"])

    def test_first_comment_never_breaks_the_run(self):
        from reelbot import publish
        with mock.patch("reelbot.publish.requests.post", side_effect=RuntimeError("down")):
            self.assertIsNone(publish.post_comment(Config(), "123", "hello"))


class Requested(unittest.TestCase):
    def test_every_caption_ends_with_the_six_tags(self):
        script, _ = run_attempt(SHORT, SHORT_REVIEW, short_plan())
        cap = script_gen.build_caption(script)
        self.assertTrue(cap.endswith("#fyp #doubletap #growthmindset #viral #reels #explorepage"))
        self.assertIn("AI-generated video.", cap)

    def test_preferred_hook_styles_win_ties(self):
        raw = dict(SHORT, hook_options=["You are asking your assistant the wrong question.", "Nobody mentions this about your prompts."],
                   hook_mechanisms=["wrong way", "nobody mentions"])
        self.assertTrue(script_gen.finalize(raw, short_plan())["hook"].startswith("Nobody mentions"))

    def test_brand_lines_are_short(self):
        from reelbot import brand
        self.assertLessEqual(len(brand.INTRO.split()), 11)
        self.assertLessEqual(len(brand.OUTRO_SHORT.split()), 6)
