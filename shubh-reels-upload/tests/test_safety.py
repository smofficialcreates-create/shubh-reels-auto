import unittest

from reelbot import safety
from reelbot.sample import SAMPLE


class SafetyTests(unittest.TestCase):
    def test_sample_passes_except_disclosure(self):
        issues = safety.check_script(dict(SAMPLE))
        self.assertEqual([i for i in issues if "disclosure" not in i], [])

    def test_blocks_price_and_stats(self):
        s = dict(SAMPLE, sentences=SAMPLE["sentences"][:-1] + ["It costs $20 a month and boosts output 40%."])
        issues = " ".join(safety.check_script(s))
        self.assertIn("price", issues)
        self.assertIn("statistic", issues)

    def test_blocks_tobacco_and_alcohol(self):
        self.assertTrue(safety.check_text("x", "Pour a whisky and smoke a cigarette."))

    def test_skills_is_not_flagged(self):
        self.assertEqual(safety.check_text("x", "Career skills compound over time."), [])

    def test_similarity(self):
        self.assertIsNotNone(safety.too_similar("batching email into set windows", ["batching email and messages into set windows"]))


if __name__ == "__main__":
    unittest.main()
