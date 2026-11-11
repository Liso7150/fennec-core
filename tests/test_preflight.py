import hermetic  # noqa: F401

import json
import tempfile
import unittest
from pathlib import Path

from fennec_core import preflight


class KvGeometryTests(unittest.TestCase):
    def test_reads_the_text_config_and_multiplies_key_and_value(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(json.dumps({
                "text_config": {"num_hidden_layers": 4, "num_key_value_heads": 2, "head_dim": 8},
            }), encoding="utf-8")
            # 2 (K and V) * 4 layers * 2 heads * 8 dims * 2 bytes
            self.assertEqual(preflight.kv_bytes_per_token(root), 256)

    def test_derives_head_dim_from_hidden_size(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(json.dumps({
                "num_hidden_layers": 2, "num_attention_heads": 4, "hidden_size": 32,
            }), encoding="utf-8")
            self.assertEqual(preflight.kv_bytes_per_token(root), 2 * 2 * 4 * 8 * 2)

    def test_missing_config_is_unknown_not_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNone(preflight.kv_bytes_per_token(Path(directory)))


class PlanTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "model.safetensors").write_bytes(b"x" * 1000)
        (self.root / "config.json").write_text(json.dumps({
            "num_hidden_layers": 1, "num_key_value_heads": 1, "head_dim": 1,
        }), encoding="utf-8")

    def test_weights_in_vram(self):
        report = preflight.plan(self.root, ctx=10, kv_bytes=10,
                                vram_bytes=5000, ram_available_bytes=0)
        self.assertGreaterEqual(report["weights_bytes"], 1000)
        self.assertEqual(report["kv_bytes"], 100)
        self.assertEqual(report["weights_fit"], "vram")
        self.assertTrue(report["kv_fits"])

    def test_weights_need_the_host_pool(self):
        report = preflight.plan(self.root, ctx=10, kv_bytes=10,
                                vram_bytes=500, ram_available_bytes=1000)
        self.assertEqual(report["weights_fit"], "hybrid")

    def test_weights_stream_from_disk_when_memory_is_short(self):
        report = preflight.plan(self.root, ctx=10, kv_bytes=10,
                                vram_bytes=100, ram_available_bytes=100)
        self.assertEqual(report["weights_fit"], "disk")

    def test_kv_that_does_not_fit_is_called_out(self):
        report = preflight.plan(self.root, ctx=1000, kv_bytes=10,
                                vram_bytes=500, ram_available_bytes=100)
        self.assertFalse(report["kv_fits"])
        self.assertTrue(any("KV cache" in note for note in report["notes"]))

    def test_missing_model_is_reported_not_invented(self):
        report = preflight.plan(self.root / "absent", ctx=10, kv_bytes=10,
                                vram_bytes=5000, ram_available_bytes=0)
        self.assertEqual(report["weights_fit"], "unknown")
        self.assertTrue(any("missing" in note for note in report["notes"]))

    def test_human_lines_cover_every_fact(self):
        text = "\n".join(preflight.human(preflight.plan(
            self.root, ctx=10, kv_bytes=10, vram_bytes=5000, ram_available_bytes=0)))
        for label in ("model", "weights", "KV per token", "KV at ctx", "weights fit", "KV fits"):
            self.assertIn(label, text)


if __name__ == "__main__":
    unittest.main()
