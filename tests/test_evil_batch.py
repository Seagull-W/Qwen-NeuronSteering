"""Check paired report semantics without downloading or loading a model."""

import json
from pathlib import Path
import tempfile
import unittest

from examples.evil_batch import compare, write_report


class BatchReportTests(unittest.TestCase):
    def test_change_detection_and_report_preserves_escaped_text(self):
        self.assertFalse(compare("same", "same")["text_changed"])
        self.assertTrue(compare("same", "different")["text_changed"])
        row = {"id": "one", "category": "test", "prompt": "<script>alert(1)</script>",
               "baseline": "你好\n<before>", "steered": "你好\n<after>",
               "baseline_tokens": 3, "steered_tokens": 4,
               **compare("你好\n<before>", "你好\n<after>")}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            write_report(output, {}, [row])
            data = json.loads((output / "results.json").read_text(encoding="utf-8"))
            self.assertEqual(data["summary"]["changed"], 1)
            self.assertEqual(data["results"][0]["baseline"], row["baseline"])
            html = (output / "report.html").read_text(encoding="utf-8")
            self.assertNotIn("<script>", html)
            self.assertIn("&lt;before&gt;", html)
            self.assertIn("&lt;after&gt;", html)

    def test_builtin_cases_are_unique_and_diverse(self):
        cases = json.loads(Path("examples/evil_prompts.json").read_text(encoding="utf-8"))
        self.assertGreaterEqual(len(cases), 16)
        self.assertEqual(len({case["id"] for case in cases}), len(cases))
        self.assertGreaterEqual(len({case["category"] for case in cases}), 8)
