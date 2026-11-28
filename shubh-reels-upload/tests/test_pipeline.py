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
        {"from": 1, "to": 2, "label": "the demo", "visual": {"type": "chat", "prompt": "What question should I really be asking about my career switch?",
            "result": "Ask which skills transfer, which gaps matter most, and who hires for both."}},
        {"from": 3, "to": 5, "label": "weak vs better", "visual": {"type": "compare",
            "left": {"title": "weak way", "rows": ["first thought", "vague answer"]},
            "right": {"title": "better way", "rows": ["sharper question", "specific answer"]}}},
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
        no_chat = dict(SHORT, scenes=[
            {"from": 1, "to": 2, "label": "x", "visual": {"type": "definition", "term": "Sharper", "meaning": "a question with a clear goal"}}] + SHORT["scenes"][1:])
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


class Growth(unittest.TestCase):
    def rows(self, n=24):
        import random
        r = random.Random(1)
        out = []
        for i in range(n):
            cat = ["practical_ai_tutorials", "productivity_methods", "career_skills"][i % 3]
            fmt = "short" if i % 2 else "long"
            good = cat == "practical_ai_tutorials"
            out.append({"media_id": str(i), "topic": f"t{i}", "category": cat, "hook": "h", "hook_mechanism": "nobody mentions" if good else "wrong way",
                        "format": fmt, "series": "S", "bucket": "evening", "reach": (300 if good else 100) * (2 if fmt == "short" else 1),
                        "views": 0, "watch_s": 5.0, "shares": 1 if good else 0, "saved": 1, "comments": 0, "likes": 3, "score": 30.0 if good else 5.0})
        return out

    def test_levels_and_no_change_when_data_is_thin(self):
        from reelbot import growth
        st = growth.analyze(self.rows(5))
        self.assertEqual(st["level"], "collecting")
        self.assertEqual(st["prefer_categories"], [])
        self.assertIsNone(st["format_winner"])

    def test_reliable_strategy_finds_winners(self):
        from reelbot import growth
        st = growth.analyze(self.rows(24))
        self.assertEqual(st["level"], "reliable")
        self.assertEqual(st["prefer_categories"][0], "practical_ai_tutorials")
        self.assertEqual(st["format_winner"], "short")
        self.assertIn("nobody mentions", st["prefer_hooks"])

    def test_join_ignores_reels_without_numbers(self):
        from reelbot import growth
        hist = [{"media_id": "a", "topic": "x"}, {"media_id": "b", "topic": "y", "posted_at": "2026-11-14T19:30+05:30"}]
        data = {"media": {"a": {"metrics": {}}, "b": {"metrics": {"reach": 200, "shares": 2, "saved": 3, "likes": 4, "comments": 1,
                                                                 "ig_reels_avg_watch_time": 6200}}}}
        rows = growth.join(hist, data)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["bucket"], "evening")
        self.assertEqual(rows[0]["score"], round((2 * 3 + 3 * 2 + 1 * 2 + 4) / 200 * 1000, 1))

    def test_strategy_steers_the_writer_only_when_enough_data(self):
        import os, tempfile
        from reelbot import growth
        d = tempfile.mkdtemp()
        p = os.path.join(d, "s.json")
        json.dump(growth.analyze(self.rows(24)), open(p, "w"))
        self.assertEqual(script_gen.load_strategy(p)["format_winner"], "short")
        json.dump(growth.analyze(self.rows(4)), open(p, "w"))
        self.assertEqual(script_gen.load_strategy(p), {})

    def test_collector_survives_a_rejected_metric(self):
        from reelbot import insights
        calls = []

        def fake(cfg, path, params=None):
            calls.append(params)
            if params and "," in params.get("metric", ""):
                raise insights.InsightsError("bad metric")
            if params and params["metric"] == "views":
                raise insights.InsightsError("nope")
            return {"data": [{"name": params["metric"], "values": [{"value": 7}]}]}
        with mock.patch("reelbot.insights._get", side_effect=fake):
            m = insights.media_metrics(Config(), "1")
        self.assertEqual(m["reach"], 7)
        self.assertNotIn("views", m)


class SeriesLoopCarousel(unittest.TestCase):
    def test_series_numbering(self):
        from reelbot import series
        s = series.assign({"category": "practical_ai_tutorials"}, [{"series": "PROMPT FIX"}, {"series": "FOCUS LAB"}, {"series": "PROMPT FIX"}])
        self.assertEqual((s["series"], s["episode"], series.label(s)), ("PROMPT FIX", 3, "PROMPT FIX #03"))

    def test_loop_line_ends_the_reel_and_is_inside_the_cta_section(self):
        from reelbot import brand, tts
        script, _ = run_attempt(dict(SHORT, loop_line="And that is exactly why"), SHORT_REVIEW, short_plan())
        lines = brand.spoken(script)
        self.assertEqual(lines[-1], "And that is exactly why")
        self.assertEqual(lines[-2], brand.OUTRO_SHORT)
        secs = planner.build_sections(script, tts.synth_fake(lines))
        self.assertEqual(secs[-1].i1, len(lines) - 1)

    def test_bad_loop_line_is_dropped(self):
        script, _ = run_attempt(dict(SHORT, loop_line="Is this the end of 3 things?"), SHORT_REVIEW, short_plan())
        self.assertEqual(script["loop_line"], "")

    def test_carousel_renders_and_picks_the_best_saved_reel(self):
        import tempfile
        from reelbot import carousel
        script, _ = run_attempt(SHORT, SHORT_REVIEW, short_plan())
        script.update(series="PROMPT FIX", episode=1, media_id="m1", date="2026-11-10")
        paths = carousel.render_slides(script, tempfile.mkdtemp())
        self.assertEqual(len(paths), len(script["scenes"]) + 2)
        self.assertIn("#fyp", carousel.carousel_caption(script))
        other = dict(script, media_id="m2", date="2026-11-12")
        self.assertEqual(carousel.pick_script([script, other], [], None)["media_id"], "m2")
        ins = {"media": {"m1": {"metrics": {"reach": 100, "saved": 9, "shares": 3}}, "m2": {"metrics": {"reach": 100}}}}
        self.assertEqual(carousel.pick_script([script, other], [], ins)["media_id"], "m1")
        self.assertEqual(carousel.pick_script([script, other], ["m1", "m2"], None), None)

    def test_carousel_publish_sequence(self):
        from reelbot import carousel
        cfg = Config(); cfg.ig_user_id, cfg.ig_token = "U", "T"
        posts = []

        class R:
            def __init__(s, d): s.d = d; s.status_code = 200
            def json(s): return s.d

        def post(url, **k):
            posts.append((url, k["data"]))
            return R({"id": f"id{len(posts)}"})
        with mock.patch("reelbot.carousel.requests.post", side_effect=post), \
             mock.patch("reelbot.carousel.requests.get", return_value=R({"status_code": "FINISHED"})):
            out = carousel.publish_carousel(cfg, ["u1", "u2"], "cap")
        self.assertEqual(out["id"], "id4")
        self.assertEqual(posts[2][1]["children"], "id1,id2")
        self.assertEqual(posts[2][1]["media_type"], "CAROUSEL")


class Engagement(unittest.TestCase):
    def test_styles_rotate_without_back_to_back_repeats(self):
        from reelbot import engage
        hist = []
        for i in range(24):
            st = engage.pick_style(hist, dt.date(2026, 11, 20) + dt.timedelta(days=i), keyword_ok=True)
            if hist:
                self.assertNotEqual(st, hist[-1]["cta_style"])
            hist.append({"cta_style": st})
        styles = [h["cta_style"] for h in hist]
        self.assertGreaterEqual(styles.count("keyword"), 4)
        self.assertLessEqual(styles.count("keyword"), 8)
        self.assertGreaterEqual(len(set(styles)), 6)
        self.assertNotIn("keyword", [engage.pick_style(hist, dt.date(2026, 12, 1), keyword_ok=False) for _ in range(3)])

    def test_each_style_is_checked(self):
        from reelbot import engage
        kw = {"keyword": "PROMPT", "reward_text": "Here is what I promised from the reel: ask the assistant for the question behind your question, then answer the best one and ask again for a sharper result."}
        self.assertEqual(engage.check_cta("choice", "Which would you use first, the weak one or the sharp one?", {}), [])
        self.assertTrue(engage.check_cta("choice", "Which would you use first?", {}))
        self.assertTrue(engage.check_cta("rate", "Tell me how useful this was.", {}))
        self.assertEqual(engage.check_cta("rate", "Rate this from one to five in the comments.", {}), [])
        self.assertEqual(engage.check_cta("share", "Send this to a friend who asks vague questions.", {}), [])
        self.assertEqual(engage.check_cta("keyword", "Comment PROMPT and I will send you the exact prompt.", kw), [])
        self.assertTrue(engage.check_cta("keyword", "Comment below for more.", kw))
        self.assertTrue(engage.check_cta("keyword", "Comment PROMPT and I will send it.", dict(kw, reward_text="Buy now at https://x.com")))

    def test_keyword_reel_end_to_end(self):
        raw = dict(SHORT, keyword="prompt", reward_text="Here is what I promised from the reel: ask the assistant what question you should really be asking, "
                   "pick the sharpest one it lists, then ask that one and request a specific example in your own words.",
                   cta="Comment PROMPT and I will send you the exact prompt.")
        plan = dict(short_plan(), cta_style="keyword")
        script, issues = run_attempt(raw, SHORT_REVIEW, plan)
        self.assertEqual(issues, [])
        self.assertEqual((script["keyword"], script["cta_style"]), ("PROMPT", "keyword"))
        self.assertIn("Comment PROMPT below", script["first_comment"])

    def test_teaser_changes_outro_and_caption(self):
        from reelbot import brand
        script, _ = run_attempt(SHORT, SHORT_REVIEW, dict(short_plan(), teaser="Why your assistant forgets"))
        self.assertEqual(brand.spoken(script)[-1], "I'm Shubh. Tomorrow: Why your assistant forgets. Follow so you don't miss it.")
        self.assertIn("Tomorrow: Why your assistant forgets", script_gen.build_caption(script))

    def test_teaser_schedule_and_promise_flow(self):
        from reelbot import engage
        pick = lambda h, d: {"title": "Why your assistant forgets", "category": "x"}   # noqa: E731
        hist = [{"cta_style": "question"}] * 5
        t = engage.pick_teaser(hist, dt.date(2026, 11, 20), pick, keyword_style=False)
        self.assertEqual(t, "Why your assistant forgets")
        self.assertIsNone(engage.pick_teaser(hist, dt.date(2026, 11, 20), pick, keyword_style=True))
        hist.append({"promise": t})
        self.assertEqual(engage.pending_promise(hist), t)
        eng = engage.plan_for(Config(), hist, dt.date(2026, 11, 21), "short", pick)
        self.assertEqual(eng["promised"], t)
        self.assertIsNone(eng["teaser"])
        hist.append({"cta_style": "question", "fulfills": t})
        self.assertIsNone(engage.pending_promise(hist))
        self.assertIsNone(engage.pick_teaser(hist, dt.date(2026, 11, 22), pick, keyword_style=False))

    def test_long_reels_never_get_keyword_or_teaser(self):
        from reelbot import engage
        eng = engage.plan_for(Config(), [{}] * 6, dt.date(2026, 11, 20), "long", lambda h, d: {"title": "A topic"})
        self.assertEqual((eng["cta_style"], eng["teaser"]), ("question", None))

    def test_responder_replies_once_and_skips_own_comment(self):
        from reelbot import responder, insights
        script = {"media_id": "m1", "keyword": "PROMPT", "reward_text": "Here is what I promised.", "first_comment": "Comment PROMPT below and I will send it to you.",
                  "date": "2026-11-20"}
        comments = {"data": [{"id": "c1", "text": "prompt please", "username": "fan"}, {"id": "c2", "text": "nice reel", "username": "fan2"},
                             {"id": "c3", "text": "Comment PROMPT below and I will send it to you.", "username": "me"},
                             {"id": "c4", "text": "PROMPT!!", "username": "fan3"}]}
        sent = []
        with mock.patch("reelbot.responder.insights._get", return_value=comments):
            replied = {"c4": {"status": "sent"}}
            n, notes = responder.run(Config(), [script], replied, lambda cfg, cid, text: sent.append(cid), owner="me")
            n2, _ = responder.run(Config(), [script], replied, lambda cfg, cid, text: sent.append(cid), owner="me")
        self.assertEqual((n, n2, sent), (1, 0, ["c1"]))
        self.assertEqual(replied["c1"]["status"], "sent")

    def test_responder_stops_on_permission_error_and_window(self):
        from reelbot import responder
        script = {"media_id": "m1", "keyword": "GUIDE", "reward_text": "x" * 30, "date": "2026-11-20"}
        comments = {"data": [{"id": f"c{i}", "text": "guide", "username": "u"} for i in range(3)]}

        def boom(cfg, cid, text):
            raise RuntimeError("HTTP 400 code=10 permission")
        with mock.patch("reelbot.responder.insights._get", return_value=comments):
            n, notes = responder.run(Config(), [script], {}, boom)
        self.assertEqual(n, 0)
        self.assertTrue(any("messages permission" in x for x in notes))
        import tempfile, os
        d = tempfile.mkdtemp()
        json.dump(dict(script, date="2026-11-01"), open(os.path.join(d, "a.json"), "w"))
        json.dump(script, open(os.path.join(d, "b.json"), "w"))
        self.assertEqual(len(responder.due_scripts(d, dt.date(2026, 11, 22))), 1)

    def test_growth_tracks_closing_style(self):
        from reelbot import growth
        rows = Growth().rows(24)
        for i, r in enumerate(rows):
            r["cta_style"] = "keyword" if i % 3 == 0 else "question"
            r["score"] = 40.0 if r["cta_style"] == "keyword" else 5.0
        st = growth.analyze(rows)
        self.assertEqual(st["prefer_cta"][0], "keyword")


class ResultsFirst(unittest.TestCase):
    def test_chat_demo_must_open_the_reel(self):
        late = dict(SHORT, scenes=[SHORT["scenes"][1], SHORT["scenes"][0], SHORT["scenes"][2]])
        late["scenes"] = [dict(late["scenes"][0], **{"from": 1, "to": 2}), dict(late["scenes"][1], **{"from": 3, "to": 5}), late["scenes"][2]]
        _s, issues = run_attempt(late, SHORT_REVIEW, short_plan())
        self.assertTrue(any("results first" in i or "chat" in i for i in issues))

    def test_demo_is_run_through_the_real_assistant(self):
        script, _ = run_attempt(SHORT, SHORT_REVIEW, short_plan())
        with mock.patch("reelbot.script_gen.llm.chat_json", return_value={"answer": "Ask which skills transfer, which gaps matter, and which employers hire for both."}):
            self.assertTrue(script_gen.demo_run(Config(), script))
        self.assertTrue(script["demo_real"])
        self.assertTrue(script["scenes"][0]["visual"]["result"].startswith("Ask which skills transfer, which gaps matter"))

    def test_bad_or_failed_demo_keeps_the_written_result(self):
        script, _ = run_attempt(SHORT, SHORT_REVIEW, short_plan())
        before = script["scenes"][0]["visual"]["result"]
        for bad in ({"answer": "See https://example.com now please thanks"}, {"answer": "ok"}, {"answer": "**bold** markdown answer with enough words here"}):
            with mock.patch("reelbot.script_gen.llm.chat_json", return_value=bad):
                self.assertFalse(script_gen.demo_run(Config(), script))
        with mock.patch("reelbot.script_gen.llm.chat_json", side_effect=RuntimeError("down")):
            self.assertFalse(script_gen.demo_run(Config(), script))
        self.assertEqual(script["scenes"][0]["visual"]["result"], before)
        self.assertNotIn("demo_real", script)


HI = dict(SHORT,
          hook_options=["Aap AI se galat sawaal pooch rahe ho.", "Aapka pehla prompt kaafi nahi hota.", "AI ke jawaab vague kyun aate hain?"],
          sentences=["Zyada log pehla sawaal hi AI se pooch lete hain.", "Wo sawaal aksar andaaza hota hai.", "Isliye AI bhi andaaza hi lautaata hai.",
                     "Ab ye try karo: AI se pucho ki aapko kaun sa sawaal puchna chahiye.", "Wo kuch behtar sawaal dega aur har sawaal ki wajah bhi batayega.",
                     "Sabse accha wala pucho aur jawaab specific ho jaayega."],
          cta="Kal AI se sabse pehle aap kya poochoge?", first_comment="Aap kaun sa sawaal sharp karoge? Neeche batao.",
          caption="Aap AI se galat sawaal pooch rahe ho.\nBehtar sawaal pucho.\nai prompts, productivity\nAI-generated video.",
          speech={"hook_options": ["आप एआई से गलत सवाल पूछ रहे हो।", "आपका पहला प्रॉम्प्ट काफ़ी नहीं होता।", "एआई के जवाब वेग क्यों आते हैं?"],
                  "sentences": ["ज़्यादा लोग पहला सवाल ही एआई से पूछ लेते हैं।", "वो सवाल अक्सर अंदाज़ा होता है।", "इसलिए एआई भी अंदाज़ा ही लौटाता है।",
                                "अब ये ट्राई करो: एआई से पूछो कि आपको कौन सा सवाल पूछना चाहिए।", "वो कुछ बेहतर सवाल देगा और वजह भी बताएगा।",
                                "सबसे अच्छा वाला पूछो और जवाब स्पेसिफ़िक हो जाएगा।"],
                  "cta": "कल एआई से सबसे पहले आप क्या पूछोगे?", "loop_line": ""})


def hi_plan():
    return dict(short_plan(), lang="hi")


class Hinglish(unittest.TestCase):
    def test_hinglish_script_passes_and_keeps_two_scripts(self):
        script, issues = run_attempt(HI, SHORT_REVIEW, hi_plan())
        self.assertEqual(issues, [])
        self.assertEqual(script["lang"], "hi")
        self.assertEqual(script["speech"]["hook"], HI["speech"]["hook_options"][0] if script["hook"] == HI["hook_options"][0] else script["speech"]["hook"])
        from reelbot import brand
        shown, said = brand.spoken(script), brand.speech(script)
        self.assertEqual(len(shown), len(said))
        self.assertEqual(shown[-1], brand.OUTRO_SHORT_HI)
        self.assertTrue(all(any("ऀ" <= ch <= "ॿ" for ch in line) for line in said))

    def test_speech_must_mirror_and_be_safe(self):
        short_speech = dict(HI, speech=dict(HI["speech"], sentences=HI["speech"]["sentences"][:-1]))
        _s, issues = run_attempt(short_speech, SHORT_REVIEW, hi_plan())
        self.assertTrue(any("speech has" in i for i in issues))
        latin = dict(HI, speech=dict(HI["speech"], cta="Kal AI se pehle kya poochoge?"))
        _s, issues = run_attempt(latin, SHORT_REVIEW, hi_plan())
        self.assertTrue(any("Devanagari" in i for i in issues))
        digits = dict(HI, speech=dict(HI["speech"], cta="कल 5 सवाल पूछोगे?"))
        _s, issues = run_attempt(digits, SHORT_REVIEW, hi_plan())
        self.assertTrue(any("digits" in i for i in issues))
        claim = dict(HI, sentences=HI["sentences"][:-1] + ["Ye tarika muft aur guaranteed hai."])
        _s, issues = run_attempt(claim, SHORT_REVIEW, hi_plan())
        self.assertTrue(any("claim" in i for i in issues))

    def test_voice_reads_devanagari_but_words_are_timed_on_the_hinglish_lines(self):
        from reelbot import brand, main, tts
        script, _ = run_attempt(HI, SHORT_REVIEW, hi_plan())
        nar = main.make_voice(Config(), script, fake=True)
        self.assertEqual(len(nar.sentence_times), len(brand.spoken(script)))
        secs = planner.build_sections(script, nar)
        self.assertEqual(secs[0].kind, "hook")
        self.assertEqual(secs[-1].kind, "cta")

    def test_language_selection(self):
        d0, d1 = dt.date(2026, 12, 2), dt.date(2026, 12, 3)
        self.assertEqual(script_gen.pick_lang(Config(reel_lang="en"), d0, "short"), "en")
        self.assertEqual(script_gen.pick_lang(Config(reel_lang="hi"), d0, "short"), "hi")
        self.assertEqual(script_gen.pick_lang(Config(reel_lang="hi"), d0, "long"), "en")
        self.assertEqual({script_gen.pick_lang(Config(reel_lang="alternate"), d, "short") for d in (d0, d1)}, {"en", "hi"})

    def test_hinglish_reels_have_no_keyword_or_teaser(self):
        from reelbot import engage
        cfg = Config(); cfg.ig_token = "t"
        eng = engage.plan_for(cfg, [{}] * 8, dt.date(2026, 12, 2), "short", lambda h, d: {"title": "A topic"}, None, "hi")
        self.assertNotEqual(eng["cta_style"], "keyword")
        self.assertIsNone(eng["teaser"])
        from reelbot import engage as e
        self.assertEqual(e.check_cta("choice", "Kaun sa pehle try karoge, weak ya sharp prompt?", {}), [])
        self.assertEqual(e.check_cta("rate", "Ise ek se paanch mein rate karo.", {}), [])

    def test_hindi_failure_falls_back_to_english_unless_strict(self):
        bad = dict(HI, speech=dict(HI["speech"], sentences=[]))

        def fake(cfg, system, user, temperature=0.0):
            if system == script_gen.REVIEW_SYSTEM:
                return SHORT_REVIEW
            return bad if "HINGLISH" in system else SHORT
        script_gen._NEAR.clear()
        with mock.patch("reelbot.script_gen.build_plans", return_value=[explainer_plan()]), \
             mock.patch("reelbot.llm.chat_json", side_effect=fake), mock.patch("reelbot.script_gen.demo_run", return_value=False), \
             mock.patch("reelbot.script_gen.engage.plan_for", return_value={"cta_style": "question", "teaser": None, "promised": None}):
            out = script_gen.generate(Config(reel_lang="hi"), [], dt.date(2026, 12, 2))
        self.assertEqual(out.get("lang"), "en")
        self.assertEqual(out["mode"], "explainer")
        script_gen._NEAR.clear()
        cfg = Config(reel_lang="hi")
        cfg.hi_strict = True
        with mock.patch("reelbot.script_gen.build_plans", return_value=[explainer_plan()]), \
             mock.patch("reelbot.llm.chat_json", side_effect=fake):
            with self.assertRaises(RuntimeError):
                script_gen.generate(cfg, [], dt.date(2026, 12, 2))

    def test_growth_picks_a_language_winner(self):
        from reelbot import growth
        rows = Growth().rows(24)
        for i, r in enumerate(rows):
            r["lang"] = "hi" if i % 2 else "en"
            r["reach"] = 400 if r["lang"] == "hi" else 100
        st = growth.analyze(rows)
        self.assertEqual(st["lang_winner"], "hi")
