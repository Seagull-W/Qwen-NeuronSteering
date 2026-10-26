"""Verify sweep controls, diagnostics and checkpoint reports without model weights."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from examples.evil_sweep import quality, save, variants
from neuron_steering import Target


class SweepTests(unittest.TestCase):
    def test_runner_uses_same_prompt_disables_thinking_and_checks_zero(self):
        from contextlib import contextmanager
        from examples.evil_sweep import main

        state = {"active": False, "calls": 0}

        class Inputs(dict):
            def to(self, _device):
                return self

        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                self.kwargs = kwargs
                return "fixed rendered prompt"

            def __call__(self, text, **kwargs):
                return Inputs(input_ids=torch.tensor([[7, 8]]), attention_mask=torch.ones(1, 2))

            def decode(self, ids, **kwargs):
                return "changed" if ids == [3, 4] else "baseline"

        class Model:
            generation_config = SimpleNamespace(to_dict=lambda: {})

            def to(self, device):
                return self

            def eval(self):
                return self

            def generate(self, **kwargs):
                self.last_kwargs = kwargs
                state["calls"] += 1
                return torch.tensor([[7, 8, 3, 4] if state["active"] else [7, 8, 1, 2]])

        @contextmanager
        def intervention(_layers, doses):
            state["active"] = any(doses.values())
            try:
                yield
            finally:
                state["active"] = False

        tokenizer, model = Tokenizer(), Model()
        fake_transformers = SimpleNamespace(__version__="fixture",
            AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: tokenizer),
            Qwen3_5ForConditionalGeneration=SimpleNamespace(from_pretrained=lambda *a, **k: model))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "new-run"
            argv = ["sweep", "--model", "fake", "--device", "cpu", "--limit", "1",
                    "--strengths", "0", "0.02", "--output", str(output)]
            with patch("sys.argv", argv), patch.dict("sys.modules", {"transformers": fake_transformers}), \
                 patch("examples.evil_sweep.discover_layers", return_value={}), \
                 patch("examples.evil_sweep.calibrate", side_effect=lambda m, l, ts, i, **k: {t: t.strength * 20 for t in ts}), \
                 patch("examples.evil_sweep.steer", intervention):
                main()
            data = json.loads((output / "results.json").read_text(encoding="utf-8"))
            self.assertEqual(data["metadata"]["status"], "complete")
            self.assertEqual(len(data["results"]), 4)
            self.assertEqual(state["calls"], 5)
            self.assertFalse(tokenizer.kwargs["enable_thinking"])
            self.assertTrue(data["results"][0]["zero_token_match"])
            self.assertEqual(data["results"][-1]["doses"][1]["strength"], -.02)
            with patch("sys.argv", argv), self.assertRaises(SystemExit):
                main()

    def test_ablations_share_one_zero_and_preserve_signed_targets(self):
        targets = [Target(14, 8949, 1), Target(13, 7956, -1)]
        plan = list(variants(targets, [0, .02, .05, .1, .2]))
        self.assertEqual(len(plan), 13)
        self.assertEqual(sum(m == 0 for _, m, _ in plan), 1)
        self.assertEqual(plan[0][2], targets)
        singles = [(g, m, ts) for g, m, ts in plan if len(ts) == 1]
        self.assertEqual(len(singles), 8)
        self.assertEqual({ts[0].strength for _, _, ts in singles}, {-1, 1})

    def test_quality_distinguishes_format_repetition_and_limit(self):
        valid = quality('{"tips":["a","b","c"]}', 12, 512, "format-zh")
        self.assertTrue(valid["format_pass"])
        self.assertFalse(quality('```json\n{}\n```', 12, 512, "format-zh")["format_pass"])
        self.assertFalse(quality('{"tips":"abc"}', 12, 512, "format-zh")["format_pass"])
        repetitive = quality("Repeated phrase here. " * 6, 512, 512, "neutral-en")
        self.assertTrue(repetitive["repeated_phrase_flag"])
        self.assertTrue(repetitive["at_token_limit"])
        self.assertIsNone(repetitive["format_pass"])
        self.assertTrue(quality("a\nb\nc", 3, 512, "format-en")["format_pass"])
        self.assertTrue(quality("user\nhello", 3, 512, "a")["role_marker_flag"])

    def test_checkpoint_replaces_and_records_partial_status(self):
        q = quality("a", 1, 512, "a")
        row = dict(id="a/zero/0", category="x", prompt="<script>", group="zero-control",
                   strength=0, baseline="a", steered="a", text_changed=False,
                   character_similarity=1, baseline_tokens=1, steered_tokens=1,
                   quality=q, baseline_quality=q, zero_token_match=True, behavior_review={})
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            save(output, {"status": "running"}, [])
            save(output, {"status": "failed"}, [row])
            result = json.loads((output / "results.json").read_text(encoding="utf-8"))
            self.assertEqual(result["metadata"]["status"], "failed")
            self.assertEqual(len(result["results"]), 1)
            self.assertIn("zero-control", (output / "summary.csv").read_text(encoding="utf-8-sig"))
            html = (output / "report.html").read_text(encoding="utf-8")
            self.assertIn("按神经元组与强度汇总", html)
            self.assertNotIn("<script>", html)
