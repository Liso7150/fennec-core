"""Tests for the model folder classifier and the packed-checkpoint converter.

The fixture is a miniature of the format Edge0-35B-A3B actually ships, built here from
raw bytes so every decoded value is known before the converter runs. That is the point:
the dequantisation is asserted exactly rather than statistically, and the norm dialect
shift is asserted to land on zero.
"""
import contextlib
import io
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
import shutil
import struct
import tempfile
import unittest
from unittest import mock

import hermetic  # noqa: F401
import convert


def bf16(value):
    """Round a float that brain-float represents exactly down to its 16 bits."""
    return struct.unpack("<I", struct.pack("<f", float(value)))[0] >> 16


def write_shard(path, tensors):
    """A .safetensors file from name -> (dtype, shape, bytes), headers 8-aligned."""
    table, offset, body = {}, 0, b""
    for name, (dtype, shape, blob) in tensors.items():
        pad = (-offset) % 8
        body += b"\0" * pad
        table[name] = {"dtype": dtype, "shape": list(shape),
                       "data_offsets": [offset + pad, offset + pad + len(blob)]}
        body += blob
        offset = table[name]["data_offsets"][1]
    header = json.dumps(table, separators=(",", ":")).encode("utf-8")
    header += b" " * ((-len(header)) % 8)
    path.write_bytes(struct.pack("<Q", len(header)) + header + body)


def packed_int4(stem, lead, groups, scale, bias, pattern):
    """One affine int4 matrix: 8 nibbles per U32 word, low nibble first.

    `lead` keeps the tensor's real leading dimensions - an expert stack is 3D on disk,
    which is what the shape derivation reads - and the rows are their product.
    """
    rows = 1
    for dim in lead:
        rows *= dim
    codes = (pattern * (rows * groups * 64))[: rows * groups * 64]
    words = b""
    for start in range(0, len(codes), 8):
        word = 0
        for index, code in enumerate(codes[start:start + 8]):
            word |= (code & 0xF) << (4 * index)
        words += struct.pack("<I", word)
    return {stem + ".weight": ("U32", list(lead) + [groups * 64 // 8], words),
            stem + ".scales": ("BF16", [rows, groups],
                               b"".join(struct.pack("<H", bf16(scale)) for _ in range(rows * groups))),
            stem + ".biases": ("BF16", [rows, groups],
                               b"".join(struct.pack("<H", bf16(bias)) for _ in range(rows * groups)))}


def f16(name, shape, values):
    return name, ("F16", list(shape), b"".join(struct.pack("<e", v) for v in values))


def make_packed(root: Path, gain=1.0):
    """Two layers, two experts each, hidden 64, one scale group per row."""
    tensors = {}
    for layer in range(2):
        prefix = "language_model.model.layers.%d" % layer
        tensors.update(packed_int4(prefix + ".mlp.switch_mlp.gate_proj", [2, 64], 1, 2.0, 0.5,
                                   list(range(16))))
        tensors.update(packed_int4(prefix + ".mlp.switch_mlp.up_proj", [2, 64], 1, 1.0, 0.0,
                                   [3] * 16))
        tensors.update(packed_int4(prefix + ".mlp.switch_mlp.down_proj", [2, 64], 1, 4.0, -1.0,
                                   [1, 2, 4, 8, 0, 5, 9, 15] * 2))
        # The router packs 4 uint8 per word instead of 8 nibbles.
        row = b"".join(struct.pack("<I", 0x0B0A0908) for _ in range(32))
        tensors[prefix + ".mlp.gate.weight"] = ("U32", [2, 32], row * 2)
        tensors[prefix + ".mlp.gate.scales"] = ("BF16", [2, 2], struct.pack("<4H", *[bf16(1.0)] * 4))
        tensors[prefix + ".mlp.gate.biases"] = ("BF16", [2, 2], struct.pack("<4H", *[bf16(0.0)] * 4))
        tensors[prefix + ".self_attn.q_proj.weight"] = ("F16", [64, 64],
                                                       b"".join(struct.pack("<e", i % 7)
                                                                for i in range(64 * 64)))
        # A full gain in both dialects: the engine never adds or subtracts for it.
        tensors[prefix + ".self_attn.q_norm.weight"] = ("F16", [64], struct.pack("<64e", *[3.0] * 64))
        for norm in ("input_layernorm", "post_attention_layernorm"):
            tensors["%s.%s.weight" % (prefix, norm)] = ("F16", [64],
                                                        struct.pack("<64e", *[gain] * 64))
        # conv1d ships [C, k, 1] where the HF spelling is [C, 1, k].
        tensors[prefix + ".linear_attn.conv1d.weight"] = ("F16", [64, 4, 1],
                                                          struct.pack("<256e", *[i for i in range(256)]))
        # Scalars the engine reads that are not matrices at all, stored brain-float.
        tensors[prefix + ".linear_attn.A_log"] = ("BF16", [4], struct.pack("<4H", *[bf16(-0.5)] * 4))
        tensors[prefix + ".linear_attn.dt_bias"] = ("BF16", [4], struct.pack("<4H", *[bf16(2.0)] * 4))
    # The router's low-rank delta, merged already into the weights the reference
    # artifact ships, so the converter must not carry it into the plan.
    tensors["language_model.model.layers.0.mlp.gate.lora_A"] = ("F16", [2, 8], b"\0" * 32)
    tensors["language_model.model.layers.0.mlp.gate.lora_B"] = ("F16", [8, 2], b"\0" * 32)
    # Edge0's parallel-LM side-nets, spelled outside the language model's namespace.
    tensors["layers.0.fc1.weight"] = ("F16", [4, 4], struct.pack("<16e", *([7.0] * 16)))
    tensors["language_model.model.embed_tokens.weight"] = ("F16", [16, 64],
                                                          struct.pack("<1024e", *[0.25] * 1024))
    tensors["language_model.model.norm.weight"] = ("F16", [64], struct.pack("<64e", *[gain] * 64))
    # The head is packed like every other matrix, so its parts hang off the bare stem.
    tensors.update(packed_int4("language_model.lm_head", [16], 1, 1.0, 0.0, list(range(16))))
    write_shard(root / "model-00001-of-00002.safetensors", dict(
        item for item in tensors.items() if ".layers." not in item[0]))
    write_shard(root / "model-00002-of-00002.safetensors", dict(
        item for item in tensors.items() if ".layers." in item[0]))
    (root / "config.json").write_text(json.dumps(
        {"model_type": "qwen3_5_moe", "hidden_size": 64}), encoding="utf-8")
    (root / "tokenizer.json").write_text("{}", encoding="utf-8")
    return tensors


def read_shard(path):
    return convert.safetensors_header(path)


def tensor(path, name):
    table = read_shard(path)
    entry = table[name]
    with open(path, "rb") as handle:
        handle.seek(entry["begin"])
        raw = handle.read(entry["end"] - entry["begin"])
    code = {"F16": "<e", "F32": "<f", "U32": "<I"}[entry["dtype"]]
    return entry, [struct.unpack(code, raw[i:i + struct.calcsize(code)])[0]
                   for i in range(0, len(raw), struct.calcsize(code))]


class Classifier(unittest.TestCase):
    def test_packed_weights_are_not_servable(self):
        with tempfile.TemporaryDirectory() as directory:
            make_packed(Path(directory))
            info = convert.classify(directory)
        self.assertEqual(info["kind"], "edge0")
        self.assertFalse(info["servable"])
        self.assertEqual(info["packed"], 9)

    def test_a_plain_hf_folder_is_servable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_shard(root / "model.safetensors", {
                "model.language_model.layers.0.input_layernorm.weight":
                    ("F16", [4], struct.pack("<4e", *([0.1] * 4)))})
            (root / "config.json").write_text("{}", encoding="utf-8")
            info = convert.classify(root)
        self.assertEqual(info["kind"], "hf")
        self.assertTrue(info["servable"])

    def test_container_gguf_empty_and_missing_are_named(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "container").mkdir()
            (root / "container" / "qwen36_meta.json").write_text("{}", encoding="utf-8")
            (root / "container" / "model-00001-of-00001.safetensors").write_bytes(b"\0" * 64)
            (root / "gguf").mkdir()
            (root / "gguf" / "model.gguf").write_bytes(b"GGUF")
            (root / "empty").mkdir()
            self.assertEqual(convert.classify(root / "container")["kind"], "container")
            self.assertTrue(convert.classify(root / "container")["servable"])
            self.assertEqual(convert.classify(root / "gguf")["kind"], "gguf")
            self.assertFalse(convert.classify(root / "gguf")["servable"])
            self.assertEqual(convert.classify(root / "empty")["kind"], "empty")
            self.assertEqual(convert.classify(root / "nope")["kind"], "missing")

    def test_header_of_a_truncated_or_foreign_file_is_empty(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "short.safetensors"
            path.write_bytes(b"\0\0")
            self.assertEqual(convert.safetensors_header(path), {})
            path.write_bytes(struct.pack("<Q", 99) + b"not json")
            self.assertEqual(convert.safetensors_header(path), {})
            self.assertEqual(convert.safetensors_header(Path(directory) / "absent.safetensors"), {})


class Dialect(unittest.TestCase):
    def test_gain_norms_are_detected_and_delta_norms_are_not(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_packed(root, gain=1.0)
            self.assertEqual(convert.norm_dialect(root)["dialect"], "gain")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_packed(root, gain=0.02)
            self.assertEqual(convert.norm_dialect(root)["dialect"], "delta")

    def test_a_final_norm_far_from_zero_does_not_flip_the_verdict(self):
        """The base's own final norm averages +1.63 and is still the delta dialect;
        that is why the test is the median over the block norms."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_packed(root, gain=0.02)
            table = convert.entries(root)
            name = "language_model.model.norm.weight"
            entry = table[name]
            with open(entry["path"], "r+b") as handle:
                handle.seek(entry["begin"])
                handle.write(struct.pack("<64e", *([1.63] * 64)))
            report = convert.norm_dialect(root)
        self.assertEqual(report["dialect"], "delta")
        self.assertEqual(report["block_norms"], 4)
        self.assertLess(report["median"], 0.5)

    def test_unknown_when_there_is_nothing_to_measure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_shard(root / "model.safetensors", {"a.b": ("F32", [2], struct.pack("<2f", 5.0, 5.0))})
            (root / "config.json").write_text("{}", encoding="utf-8")
            self.assertEqual(convert.norm_dialect(root)["dialect"], "unknown")


class Naming(unittest.TestCase):
    def test_target_names_follow_the_hf_spelling(self):
        self.assertEqual(convert.target_name("language_model.model.embed_tokens.weight"),
                         "model.language_model.embed_tokens.weight")
        self.assertEqual(convert.target_name("language_model.lm_head.weight"), "lm_head.weight")
        self.assertEqual(convert.target_name("model.language_model.norm.weight"),
                         "model.language_model.norm.weight")

    def test_a_packed_head_is_named_by_what_the_engine_binds(self):
        """`language_model.lm_head` is a stem, not `language_model.lm_head.weight`, so
        the respell cannot insist on the suffix."""
        self.assertEqual(convert.target_name("language_model.lm_head"), "lm_head")
        self.assertEqual(convert.target_name("language_model.lm_head.scales"), "lm_head.scales")

    def test_plan_keeps_the_vectors_and_drops_what_the_engine_never_reads(self):
        with tempfile.TemporaryDirectory() as directory:
            make_packed(Path(directory))
            table = convert.entries(directory)
            specs = convert.build_plan(table)
        # The head goes through the decoder like any other packed matrix.
        self.assertEqual(specs["lm_head.weight"],
                         ("dequant", ("language_model.lm_head",), [16, 64]))
        # A vector that is not a matrix still has to reach the artifact.
        self.assertEqual(specs["model.language_model.layers.0.linear_attn.A_log"],
                         ("copy", ("language_model.model.layers.0.linear_attn.A_log",), [4]))
        self.assertEqual(specs["model.language_model.layers.0.linear_attn.dt_bias"][0], "copy")
        # The side-nets live outside the namespace, and the LoRA pair is already merged.
        self.assertNotIn("layers.0.fc1.weight", specs)
        for name in specs:
            self.assertFalse(name.endswith((".lora_A", ".lora_B")), name)
        self.assertEqual(convert.foreign_count(table), 1)

    def test_plan_fuses_gate_and_up_once_and_renames_the_stacks(self):
        with tempfile.TemporaryDirectory() as directory:
            make_packed(Path(directory))
            table = convert.entries(directory)
            specs = convert.build_plan(table)
        experts = "model.language_model.layers.1.mlp.experts."
        self.assertEqual(specs[experts + "gate_up_proj"][0], "fuse")
        self.assertEqual(specs[experts + "gate_up_proj"][2], [2, 128, 64])
        self.assertEqual(specs[experts + "down_proj"][2], [2, 64, 64])
        self.assertEqual(specs["model.language_model.layers.1.mlp.gate.weight"][2], [2, 128])
        self.assertEqual(specs["lm_head.weight"][0], "dequant")
        self.assertEqual(specs["model.language_model.layers.1.linear_attn.conv1d.weight"][2],
                         [64, 1, 4])
        # gate and up each appear once, as halves of the fused stack, never on their own
        self.assertNotIn(experts + "gate_proj", specs)
        self.assertNotIn(experts + "up_proj", specs)
        for name in specs:
            self.assertNotIn(".scales", name)
            self.assertNotIn(".biases", name)

    def test_packing_is_read_from_the_shapes(self):
        with tempfile.TemporaryDirectory() as directory:
            make_packed(Path(directory))
            table = convert.entries(directory)
            stem = "language_model.model.layers.0.mlp.switch_mlp.gate_proj"
            self.assertEqual(convert.packing(table, stem), convert.INT4)
            self.assertEqual(convert.packing(table, "language_model.model.layers.0.mlp.gate"),
                             convert.INT8)

    def test_an_unrecognisable_packing_is_refused_not_guessed(self):
        table = {"a.weight": {"dtype": "U32", "shape": [2, 7]},
                 "a.scales": {"dtype": "BF16", "shape": [2, 3]}}
        with self.assertRaises(ValueError):
            convert.packing(table, "a")


class Plan(unittest.TestCase):
    def test_edge0_plan_reports_size_and_a_blocker_when_the_volume_is_full(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "model"
            root.mkdir()
            make_packed(root)
            planned = convert.plan(root)
            self.assertEqual(planned["kind"], "edge0")
            self.assertEqual(planned["action"], "dequant")
            self.assertEqual(planned["dialect"]["dialect"], "gain")
            self.assertGreater(planned["est_bytes"], 0)
            self.assertEqual(planned["foreign"], 1)
            self.assertIsNone(planned.get("blocker"))
            # The estimate is the decoded size, not the packed one: 2 bytes per value.
            table = convert.entries(root)
            expect = sum(convert._numel(spec[2]) * 2
                         for spec in convert.build_plan(table).values())
            self.assertEqual(planned["est_bytes"], expect)
            # An HF folder needs no conversion at all, so it gets no size estimate.
            plain = root.parent / "hf"
            plain.mkdir()
            write_shard(plain / "model.safetensors", {
                "model.language_model.layers.0.input_layernorm.weight":
                    ("F16", [4], struct.pack("<4e", *([0.1] * 4)))})
            (plain / "config.json").write_text("{}", encoding="utf-8")
            quiet = convert.plan(plain)
            self.assertEqual(quiet["action"], "none")
            self.assertNotIn("est_bytes", quiet)
            loud = dict(quiet, action="copy_norms")
            self.assertEqual(loud["action"], "copy_norms")


class Convert(unittest.TestCase):
    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        self.root = Path(self._temp.name)
        self.source = self.root / "edge0-mini"
        self.source.mkdir()
        make_packed(self.source)
        self.dest = self.root / "cache" / "converted" / "edge0-mini"

    def tearDown(self):
        try:
            self._temp.cleanup()
        except OSError:
            pass

    def test_conversion_reproduces_the_affine_decode_exactly(self):
        try:
            import numpy  # noqa: F401
        except ImportError:
            self.skipTest("numpy not installed")
        state = convert.convert(self.source, self.dest)
        self.assertEqual(state["status"], "complete")
        self.assertEqual(state["tensors"], len(convert.build_plan(convert.entries(self.source))))
        self.assertEqual(state["converter_version"], convert.VERSION)
        self.assertEqual(state["dialect"]["dialect"], "gain")
        shards = sorted(self.dest.glob("model-*.safetensors"))
        self.assertTrue(shards)
        index = json.loads((self.dest / "model.safetensors.index.json").read_text(encoding="utf-8"))
        self.assertIn("model.language_model.layers.1.mlp.experts.gate_up_proj",
                      index["weight_map"])
        self.assertEqual(index["metadata"]["total_size"],
                         sum(shard.stat().st_size for shard in shards))
        # gate carries scale 2.0 and bias 0.5, up carries 1.0 and 0.0, and the codes
        # cycle 0..15, so every decoded column is predictable without re-running the math.
        shard = self.dest / index["weight_map"][
            "model.language_model.layers.1.mlp.experts.gate_up_proj"]
        _, values = tensor(shard, "model.language_model.layers.1.mlp.experts.gate_up_proj")
        gate = [2.0 * (i % 16) + 0.5 for i in range(64)]
        up = [1.0 * 3 + 0.0] * 64
        self.assertEqual(values[:64], gate)
        self.assertEqual(values[64 * 64:64 * 64 + 64], up)

    def test_norms_leave_the_gain_dialect_and_q_norm_stays_put(self):
        try:
            import numpy  # noqa: F401
        except ImportError:
            self.skipTest("numpy not installed")
        convert.convert(self.source, self.dest)
        index = json.loads((self.dest / "model.safetensors.index.json").read_text(encoding="utf-8"))
        for name in ("model.language_model.layers.0.input_layernorm.weight",
                     "model.language_model.layers.1.post_attention_layernorm.weight",
                     "model.language_model.norm.weight"):
            _, values = tensor(self.dest / index["weight_map"][name], name)
            self.assertEqual(set(values), {0.0}, name)
        self.assertEqual(convert.norm_dialect(self.dest)["dialect"], "delta")
        kept = "model.language_model.layers.0.self_attn.q_norm.weight"
        _, values = tensor(self.dest / index["weight_map"][kept], kept)
        self.assertEqual(set(values), {3.0})

    def test_already_delta_norms_are_left_alone(self):
        try:
            import numpy  # noqa: F401
        except ImportError:
            self.skipTest("numpy not installed")
        delta = self.root / "edge0-delta"
        delta.mkdir()
        make_packed(delta, gain=0.02)
        out = self.root / "cache" / "converted" / "edge0-delta"
        convert.convert(delta, out)
        index = json.loads((out / "model.safetensors.index.json").read_text(encoding="utf-8"))
        name = "model.language_model.layers.0.input_layernorm.weight"
        _, values = tensor(out / index["weight_map"][name], name)
        kept = struct.unpack("<e", struct.pack("<e", 0.02))[0]
        self.assertEqual(set(values), {kept})

    def test_conv1d_singleton_moves_and_the_bytes_survive(self):
        try:
            import numpy  # noqa: F401
        except ImportError:
            self.skipTest("numpy not installed")
        convert.convert(self.source, self.dest)
        index = json.loads((self.dest / "model.safetensors.index.json").read_text(encoding="utf-8"))
        name = "model.language_model.layers.0.linear_attn.conv1d.weight"
        entry, values = tensor(self.dest / index["weight_map"][name], name)
        self.assertEqual(entry["shape"], [64, 1, 4])
        self.assertEqual(values[:8], [0, 1, 2, 3, 4, 5, 6, 7])

    def test_int8_router_decodes_byte_wise(self):
        try:
            import numpy  # noqa: F401
        except ImportError:
            self.skipTest("numpy not installed")
        convert.convert(self.source, self.dest)
        index = json.loads((self.dest / "model.safetensors.index.json").read_text(encoding="utf-8"))
        name = "model.language_model.layers.0.mlp.gate.weight"
        entry, values = tensor(self.dest / index["weight_map"][name], name)
        self.assertEqual(entry["shape"], [2, 128])
        self.assertEqual(values[:8], [8.0, 9.0, 10.0, 11.0, 8.0, 9.0, 10.0, 11.0])

    def test_a_brain_float_vector_becomes_a_half_one(self):
        """The engine reads one float width; A_log and dt_bias arrive brain-float and the
        reference artifact keeps them half, at the same value."""
        try:
            import numpy  # noqa: F401
        except ImportError:
            self.skipTest("numpy not installed")
        convert.convert(self.source, self.dest)
        index = json.loads((self.dest / "model.safetensors.index.json").read_text(encoding="utf-8"))
        for name, value in (("model.language_model.layers.0.linear_attn.A_log", -0.5),
                            ("model.language_model.layers.1.linear_attn.dt_bias", 2.0)):
            entry, values = tensor(self.dest / index["weight_map"][name], name)
            self.assertEqual(entry["dtype"], "F16", name)
            self.assertEqual(set(values), {value}, name)

    def test_what_the_plan_leaves_out_never_reaches_the_artifact(self):
        try:
            import numpy  # noqa: F401
        except ImportError:
            self.skipTest("numpy not installed")
        convert.convert(self.source, self.dest)
        index = json.loads((self.dest / "model.safetensors.index.json").read_text(encoding="utf-8"))
        for name in index["weight_map"]:
            self.assertFalse(name.endswith(convert.CHECKPOINT_PARTS), name)
            self.assertNotEqual(name, "layers.0.fc1.weight")

    def test_sidecars_are_the_checkpoints_own_and_absence_is_fatal(self):
        try:
            import numpy  # noqa: F401
        except ImportError:
            self.skipTest("numpy not installed")
        convert.convert(self.source, self.dest)
        self.assertEqual(json.loads((self.dest / "config.json").read_text(encoding="utf-8")),
                         json.loads((self.source / "config.json").read_text(encoding="utf-8")))
        self.assertTrue((self.dest / "tokenizer.json").is_file())
        (self.source / "tokenizer.json").unlink()
        with self.assertRaises(ValueError):
            convert.convert(self.source, self.dest)

    def test_a_folder_with_nothing_packed_is_not_converted(self):
        try:
            import numpy  # noqa: F401
        except ImportError:
            self.skipTest("numpy not installed")
        plain = self.root / "hf"
        plain.mkdir()
        write_shard(plain / "model.safetensors", {"a.weight": ("F16", [2], struct.pack("<2e", 1, 2))})
        (plain / "config.json").write_text("{}", encoding="utf-8")
        with self.assertRaises(ValueError):
            convert.convert(plain, self.root / "out")

    def test_rerun_replaces_shards_rather_than_adding_to_them(self):
        try:
            import numpy  # noqa: F401
        except ImportError:
            self.skipTest("numpy not installed")
        convert.convert(self.source, self.dest)
        (self.dest / "model-00099-of-00099.safetensors").write_bytes(b"")
        write_shard(self.dest / "model-00098-of-00098.safetensors",
                    {"ghost.weight": ("F16", [2], struct.pack("<2e", 1, 2))})
        convert.convert(self.source, self.dest)
        self.assertFalse((self.dest / "model-00098-of-00098.safetensors").exists())
        index = json.loads((self.dest / "model.safetensors.index.json").read_text(encoding="utf-8"))
        self.assertNotIn("ghost.weight", index["weight_map"])


class State(unittest.TestCase):
    def test_state_merges_and_never_leaves_a_temp_file(self):
        with tempfile.TemporaryDirectory() as directory:
            dest = Path(directory) / "out"
            convert.write_state(dest, status="running", step="a")
            convert.write_state(dest, percent=50)
            state = convert.read_state(dest)
            self.assertEqual(state["status"], "running")
            self.assertEqual(state["percent"], 50)
            self.assertIn("started_at", state)
            self.assertTrue(dest.is_dir())
            self.assertEqual(list(dest.glob("*.tmp")), [])
            self.assertIsNone(convert.read_state(Path(directory) / "absent"))

    def test_resolve_names_the_state_without_starting_a_conversion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "edge0-mini"
            source.mkdir()
            make_packed(source)
            cache = root / "cache"
            first = convert.resolve(source, cache_root=cache, start=False)
            self.assertEqual(first["state"], "pending")
            self.assertTrue(first["path"].endswith("edge0-mini"))
            # Nothing may be written into the user's source folder.
            self.assertEqual(sorted(item.name for item in source.iterdir()),
                             ["config.json", "model-00001-of-00002.safetensors",
                              "model-00002-of-00002.safetensors", "tokenizer.json"])
            convert.write_state(first["path"], status="complete", tensors=30, bytes=1,
                                seconds=0.1, source=str(source), dialect={"dialect": "gain"},
                                converter_version=convert.VERSION, finished_at="now")
            again = convert.resolve(source, cache_root=cache, start=False)
            self.assertEqual(again["state"], "ready")
            self.assertEqual(again["path"], first["path"])
            self.assertTrue(again["converted"])
            self.assertEqual(again["provenance"]["tensors"], 30)

    def test_resolve_passes_a_ready_folder_straight_through(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            container = root / "container"
            container.mkdir()
            (container / "qwen36_meta.json").write_text("{}", encoding="utf-8")
            (container / "model-globals.safetensors").write_bytes(b"\0" * 32)
            info = convert.resolve(container, cache_root=root / "cache", start=False)
            self.assertEqual(info["state"], "ready")
            self.assertEqual(Path(info["path"]), container)
            gguf = root / "gguf"
            gguf.mkdir()
            (gguf / "m.gguf").write_bytes(b"x")
            self.assertEqual(convert.resolve(gguf, start=False)["state"], "unusable")


class Cleanup(unittest.TestCase):
    """What may be deleted, and what may only be measured.

    The line the whole class defends: a conversion this module abandoned is junk, a
    model the user downloaded is not - and a finished artifact whose download has gone
    is the last copy of that model, however orphaned it looks.
    """

    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        self.root = Path(self._temp.name)
        self.models = self.root / "models"
        self.cache = self.root / "cache"
        self.converted = self.cache / "converted"
        self.models.mkdir()
        self.converted.mkdir(parents=True)

    def tearDown(self):
        try:
            self._temp.cleanup()
        except OSError:
            pass

    def entry(self, name, *, status, source, age=None, bytes_=4096):
        """A cache artifact and the source it came from, aged if asked."""
        destination = self.converted / name
        destination.mkdir(parents=True)
        (destination / "model-00001-of-00001.safetensors").write_bytes(b"\0" * bytes_)
        convert.write_state(destination, status=status, step="halfway", source=str(source))
        if age is not None:
            path = convert.state_path(destination)
            state = json.loads(path.read_text(encoding="utf-8"))
            stamp = datetime.now(timezone.utc) - timedelta(seconds=age)
            state["updated_at"] = stamp.isoformat(timespec="seconds")
            path.write_text(json.dumps(state), encoding="utf-8")
        return destination

    def test_only_a_run_that_can_be_redone_is_reclaimed(self):
        lived = self.models / "lived"
        lived.mkdir()
        gone = self.models / "gone"
        gone.mkdir()
        dead = self.entry("failed", status="failed", source=lived)
        hung = self.entry("hung", status="running", source=lived,
                          age=convert.STALE_AFTER_S + 60)
        live = self.entry("running", status="running", source=lived)
        # A complete artifact whose source is gone: nothing resolves to it, but deleting
        # it would delete the model, so it is reported and kept.
        last = self.entry("orphan", status="complete", source=gone)
        # A folder in the cache this module never wrote is not this module's to judge.
        foreign = self.converted / "foreign"
        foreign.mkdir()
        (foreign / "keepme.bin").write_bytes(b"\0" * 8)
        shutil.rmtree(gone)

        report = convert.cleanup(cache_root=self.cache)
        reclaimed = {item["name"]: item for item in report["reclaimed"]}
        self.assertEqual(set(reclaimed), {"failed", "hung"})
        # Each reclaimed folder holds its 4096-byte shard plus the state file, so only
        # the shard is counted for the failed one and both for the folder taken whole.
        self.assertEqual(report["bytes_reclaimed"],
                         sum(item["bytes"] for item in report["reclaimed"]))
        self.assertEqual(reclaimed["failed"]["bytes"], 4096)
        self.assertGreater(reclaimed["hung"]["bytes"], 4096)
        self.assertEqual(reclaimed["failed"]["scope"], "shards")
        self.assertEqual(reclaimed["hung"]["scope"], "folder")
        self.assertFalse(hung.is_dir())
        # The failed run keeps its reason: the shards go, the state file stays, so
        # `resolve` still tells the model list why this one is not servable.
        self.assertTrue(dead.is_dir())
        self.assertFalse(list(dead.glob("model-*.safetensors")))
        self.assertEqual(convert.read_state(dead)["status"], "failed")
        self.assertTrue(live.is_dir() and last.is_dir() and foreign.is_dir())
        self.assertEqual([item["name"] for item in report["orphaned"]], ["orphan"])

    def test_a_finished_conversion_sweeps_its_own_cache(self):
        try:
            import numpy  # noqa: F401
        except ImportError:
            self.skipTest("numpy not installed")
        source = self.models / "edge0-mini"
        source.mkdir()
        make_packed(source)
        stale = self.entry("stale", status="failed", source=source)
        state = convert.convert(source, self.converted / "edge0-mini")
        self.assertEqual(state["status"], "complete")
        self.assertEqual([item["name"] for item in state["reclaimed"]], ["stale"])
        self.assertEqual([item["scope"] for item in state["reclaimed"]], ["shards"])
        self.assertFalse(list(stale.glob("model-*.safetensors")))
        self.assertEqual(convert.read_state(stale)["status"], "failed")
        # The fresh artifact survives the sweep it triggered.
        self.assertTrue((self.converted / "edge0-mini" / "fennec-convert.json").is_file())

    def test_unsupported_models_are_priced_but_not_touched(self):
        packed = self.models / "edge0-mini"
        packed.mkdir()
        make_packed(packed)
        gguf = self.models / "big-gguf"
        gguf.mkdir()
        (gguf / "m.gguf").write_bytes(b"\0" * 2048)
        shutil.copytree(packed, self.models / ".hidden", dirs_exist_ok=True)

        report = convert.cleanup(models_root=self.models, cache_root=self.cache)
        # Packed is what the converter fixes, so it is never cleanup; a dot folder is
        # somebody's, and neither may appear in the list of things to remove.
        self.assertEqual([item["name"] for item in report["unsupported"]], ["big-gguf"])
        self.assertEqual(report["unsupported"][0]["bytes"], 2048)
        self.assertEqual(report["bytes_model"], 2048)
        self.assertTrue(gguf.is_dir())

        removed = convert.cleanup(models_root=self.models, cache_root=self.cache,
                                  models=True)
        self.assertTrue(removed["unsupported"][0]["removed"])
        self.assertFalse(gguf.is_dir())
        self.assertTrue(packed.is_dir())

    def test_cli_cleanup_reports_and_keeps_without_the_flag(self):
        gguf = self.models / "big-gguf"
        gguf.mkdir()
        (gguf / "m.gguf").write_bytes(b"\0" * 2048)
        out, err = io.StringIO(), io.StringIO()
        # The CLI takes its cache from the environment, so the sweep must be pinned to
        # the temp cache: left alone it would reach the real ~/.cache/fennec.
        with mock.patch.dict(os.environ, {"FENNEC_CACHE": str(self.cache)}), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            self.assertEqual(convert.main(["--cleanup", str(self.models)]), 0)
        self.assertEqual(json.loads(out.getvalue())["bytes_model"], 2048)
        self.assertIn("unsupported big-gguf", err.getvalue())
        self.assertTrue(gguf.is_dir())


class Output(unittest.TestCase):
    def test_written_header_matches_the_official_reader(self):
        try:
            import numpy
        except ImportError:
            self.skipTest("numpy not installed")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "x.safetensors"
            payload = {"a.weight": numpy.arange(6, dtype=numpy.float16).reshape(2, 3),
                       "b.norm": numpy.ones(4, dtype=numpy.float16)}
            size = convert._write_safetensors(path, payload)
            self.assertEqual(size, path.stat().st_size)
            table = convert.safetensors_header(path)
            self.assertEqual(table["a.weight"]["dtype"], "F16")
            self.assertEqual(table["a.weight"]["shape"], [2, 3])
            # Only the data section's start has to be 8-aligned; tensors follow each
            # other at consecutive offsets.
            self.assertEqual(table["a.weight"]["begin"] % 8, 0)
            self.assertEqual(table["b.norm"]["begin"], table["a.weight"]["end"])
            for name, entry in table.items():
                with open(path, "rb") as handle:
                    handle.seek(entry["begin"])
                    raw = handle.read(entry["end"] - entry["begin"])
                self.assertEqual(raw, payload[name].tobytes(), name)

    def test_shard_order_keeps_the_head_alone_and_layers_in_sequence(self):
        specs = {"model.language_model.embed_tokens.weight": 1,
                 "lm_head.weight": 1,
                 "model.language_model.layers.3.mlp.gate.weight": 1,
                 "model.language_model.layers.1.mlp.gate.weight": 1,
                 "model.language_model.norm.weight": 1}
        order = convert._shards(specs)
        self.assertEqual([label for label, _ in order],
                         ["globals", "lm_head", "layer 1", "layer 3"])
        self.assertNotIn("lm_head.weight", order[0][1])
        self.assertEqual(order[2][1], ["model.language_model.layers.1.mlp.gate.weight"])
        # A folder with only layers must not publish an empty globals shard.
        self.assertEqual([label for label, _ in convert._shards(
            {"model.language_model.layers.0.a": 1})], ["layer 0"])

    def test_cli_inspect_reports_classification_without_numpy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "edge0-mini"
            source.mkdir()
            make_packed(source)
            code = convert.main(["--inspect", str(source)])
            self.assertEqual(code, 0)


class CliArguments(unittest.TestCase):
    """A flag is not a folder. `fennec-convert --help` used to fall through to the
    SOURCE branch, so it wrote a failed state into `<cache>/converted/--help`, left the
    junk folder behind for `--cleanup` to find, and exited 1 for a request that succeeded.
    Every case here runs against an isolated cache and asserts that folder never appears."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.converted = Path(self.tmp.name) / "cache" / "converted"
        env = mock.patch.dict(os.environ, {"FENNEC_CACHE": str(self.converted.parent)})
        env.start()
        self.addCleanup(env.stop)

    def run_main(self, argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = convert.main(argv)
        return code, out.getvalue(), err.getvalue()

    def test_help_answers_on_stdout_and_writes_nothing(self):
        for flag in ("-h", "--help"):
            code, out, err = self.run_main([flag])
            self.assertEqual(code, 0, "%s is a request that succeeded" % flag)
            self.assertIn("usage:", out)
            self.assertEqual(err, "", "%s complained on stderr" % flag)
            self.assertFalse(self.converted.exists(), "%s reached the path branch" % flag)

    def test_an_unknown_option_is_refused_before_a_folder_is_made(self):
        code, out, err = self.run_main(["--dequantise", "edge0-35b"])
        self.assertEqual(code, 2, "a bad flag is a usage error, not a failed conversion")
        self.assertIn("unknown option --dequantise", err)
        self.assertFalse(self.converted.exists(),
                         "the flag was taken for a model folder and its state written")

    def test_no_arguments_asks_for_usage_without_touching_the_cache(self):
        code, out, err = self.run_main([])
        self.assertEqual(code, 2)
        self.assertIn("usage:", err)
        self.assertFalse(self.converted.exists())

    def test_a_folder_named_after_a_flag_is_still_a_path(self):
        # The guard keys on the first argument only, so `--` and a real folder whose
        # name begins with a dash stay reachable rather than becoming unreachable.
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "-weird"
            source.mkdir()
            code, out, err = self.run_main([str(source)])
            self.assertEqual(code, 1, "an empty folder has nothing to dequantise")
            self.assertIn("nothing to dequantise", err)


class EngineSeam(unittest.TestCase):
    """The engine must never receive the packed folder, only what `resolve` returns."""

    def setUp(self):
        from types import SimpleNamespace
        import engine
        self.engine = engine
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        self.models = self.root / "models"
        self.models.mkdir()
        self.cache = self.root / "cache"
        self.source = self.models / "edge0-mini"
        self.source.mkdir()
        make_packed(self.source)
        self.config = SimpleNamespace(model=str(self.source), cache_root=self.cache,
                                      policy="quality", ctx=0, backend="auto",
                                      engine_root=self.root, api_key="local")
        self.stub = SimpleNamespace(config=self.config)
        self.stub.model_path = lambda *a, **k: engine.Engine.model_path(self.stub, *a, **k)
        self.stub._model_args = lambda **k: engine.Engine._model_args(self.stub, **k)

    def test_a_packed_folder_is_refused_rather_than_served_as_nonsense(self):
        # start=False: a diagnostic must not launch a multi-hour conversion as a side effect
        with self.assertRaises(self.engine.EngineError) as caught:
            self.stub.model_path()
        self.assertIn("not servable", str(caught.exception))
        self.assertIn("edge0-mini", str(caught.exception))

    def test_a_completed_conversion_becomes_the_model_path(self):
        destination = convert.output_for(self.source, self.cache)
        destination.mkdir(parents=True)
        convert.write_state(destination, status="complete", tensors=10)
        self.assertEqual(self.stub.model_path(), str(destination))
        self.assertEqual(self.stub._model_args(),
                         ["--model", str(destination), "--policy", "quality", "--ctx", "0"])
        # FENNEC_MODEL must agree with --model, or the child reads the packed path
        engine = self.engine
        env = engine.Engine._engine_env(self.stub, model=str(destination))
        self.assertEqual(env["FENNEC_MODEL"], str(destination))

    def test_conversion_progress_reaches_status_without_a_full_resolve(self):
        destination = convert.output_for(self.source, self.cache)
        destination.mkdir(parents=True)
        convert.write_state(destination, status="running", step="layer 3", percent=42)
        state = self.engine.Engine._convert_state(self.stub)
        self.assertEqual(state, {"status": "running", "step": "layer 3", "percent": 42,
                                 "error": None})
        # A finished artifact leaves no progress in status
        convert.write_state(destination, status="complete")
        self.assertIsNone(self.engine.Engine._convert_state(self.stub))


if __name__ == "__main__":
    unittest.main()
