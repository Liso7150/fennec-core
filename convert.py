"""What a model folder really holds, and the conversion needed to serve it.

`scan_models` used to call a folder ready when a config.json and a .safetensors sat
next to each other. Edge0-35B-A3B satisfies that and cannot be served at all: every
quantized matrix ships as <stem>.weight (U32) plus <stem>.scales and <stem>.biases
(BF16), and the experts are three separate tensors where the engine wants one fused
stack. Handing that to the engine raises nothing. Out of the other end comes fluent
nonsense, which is why readiness has to be judged from the tensor names rather than
from the file extensions.

Two things are read off the data, never off the folder name or the model class.

The packing. One scale per 64 inputs with 8 decoded columns per stored word is int4;
one scale per 64 with 4 decoded columns is int8 (the routers, which pack 4 uint8
values per word). w = scale * q + bias, groups of GROUP inputs.

The RMSNorm dialect. The kernel always computes (1 + w) * x, so a checkpoint that
stores the gain itself must have 1.0 subtracted on the way in. `Qwen3RMSNorm` stores a
gain, `Qwen3_5MoeRMSNorm` stores a delta, and the engine chooses between them from
`model_type` alone (D:/Fennec/src/config.rs:247). Edge0 spells itself as the second
family while shipping its norms in the first, which doubles every normalised
activation; any other mislabelled checkpoint does the same. Per-tensor thresholds
cannot catch it, because a real gain in the base's own final norm averages +1.63 while
a delta there sits near +0.07 - so the test is the median over the block norms, which
is ~0 for delta and ~1 for gain.

Reading a header is stdlib. Building the converted tensors takes numpy, which
fennec-core does not otherwise depend on, so it stays an extra and everything above
`convert()` answers questions without it.

`cleanup()` is the only function here that removes anything, and it removes only what
this module wrote: a failed or abandoned conversion whose source is still on disk. Model
folders the user downloaded get measured and named, never deleted, unless a caller
passes `models=True`.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import time

VERSION = 1
GROUP = 64
MAX_SHARDS = 512
# The tensors the kernel treats as (1 + w). q_norm, k_norm and linear_attn.norm are
# full gains in both dialects and never reach norm_w, so they stay out of this list.
NORM_SUFFIXES = ("input_layernorm.weight", "post_attention_layernorm.weight",
                 "model.language_model.norm.weight", "model.norm.weight")
SWITCH = ".mlp.switch_mlp."
EXPERTS = ".mlp.experts."
LM_HEAD = "language_model.lm_head"
# A packed matrix is `<stem>.weight` with two float sidecars beside it, and each LoRA
# pair ships as `<stem>.lora_A` / `.lora_B`. None of those pieces is a tensor of the
# artifact: the sidecars fold into `w = scale * q + bias`, and the reference checkpoint
# proves the LoRA halves are already merged into the stored weights.
CHECKPOINT_PARTS = (".scales", ".biases", ".lora_A", ".lora_B")
# The engine binds one namespace. Edge0's parallel-LM side-nets sit outside it
# unqualified (`layers.10.fc1.weight`), and the reference artifact carries none of them.
TEXT_PREFIXES = ("language_model.", "model.language_model.", "lm_head.")
SIDECARS = ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
            "generation_config.json", "special_tokens_map.json", "vocab.json", "merges.txt")
INT4, INT8 = "int4", "int8"


def safetensors_header(path):
    """One .safetensors file as name -> dtype, shape and absolute byte span."""
    path = Path(path)
    try:
        with open(path, "rb") as handle:
            span = handle.read(8)
            if len(span) != 8:
                return {}
            length = struct.unpack("<Q", span)[0]
            if not 0 < length < 64 * 1024 * 1024:
                return {}
            try:
                raw = json.loads(handle.read(length))
            except (UnicodeDecodeError, ValueError):
                return {}
    except OSError:
        return {}
    base = 8 + length
    out = {}
    if not isinstance(raw, dict):
        return {}
    for name, entry in raw.items():
        if name == "__metadata__" or not isinstance(entry, dict):
            continue
        try:
            low, high = entry["data_offsets"]
        except (KeyError, TypeError, ValueError):
            continue
        out[name] = {"dtype": entry.get("dtype"), "shape": list(entry.get("shape") or []),
                     "path": str(path), "begin": base + int(low), "end": base + int(high)}
    return out


def entries(directory) -> dict:
    """Every tensor in a folder, from the shard headers only.

    Nothing here reads weight bytes, so the whole table costs one open and one JSON
    parse per shard - a few hundred kilobytes for a 65 GB checkpoint.
    """
    root = Path(directory)
    table = {}
    for shard in sorted(root.glob("*.safetensors"))[:MAX_SHARDS]:
        table.update(safetensors_header(shard))
    return table


def packed_stems(table) -> list:
    """Tensors stored as an affine-quantized matrix: a .weight with .scales beside it."""
    return sorted(name[: -len(".weight")] for name in table
                  if name.endswith(".weight") and name[: -len(".weight")] + ".scales" in table
                  and name[: -len(".weight")] + ".biases" in table)


def tensor_names(directory):
    """Every tensor name in a folder, without reading weight bytes if it can be helped.

    The HF index lists all the keys in one small JSON, which is what a directory
    listing has to use: opening 41 shard headers per model, on every poll, is slower
    than the listing it is serving. Only `plan` and the dialect test pay for shapes.
    """
    root = Path(directory)
    index = root / "model.safetensors.index.json"
    if index.is_file():
        try:
            weight_map = json.loads(index.read_text(encoding="utf-8")).get("weight_map")
        except (OSError, ValueError):
            weight_map = None
        if isinstance(weight_map, dict) and weight_map:
            return set(weight_map)
    return set(entries(root))


def classify(directory) -> dict:
    """Name the format the folder is actually in, and whether the engine can read it."""
    root = Path(directory)
    if not root.is_dir():
        return {"kind": "missing", "servable": False, "detail": "not a directory"}
    weights = sorted(item for item in root.iterdir()
                     if item.is_file() and item.suffix.lower() in {".safetensors", ".gguf"})
    if not weights:
        return {"kind": "empty", "servable": False, "detail": "no weight files"}
    if all(item.suffix.lower() == ".gguf" for item in weights):
        return {"kind": "gguf", "servable": False,
                "detail": "GGUF is a llama.cpp container; the engine loads safetensors"}
    if (root / "qwen36_meta.json").is_file():
        return {"kind": "container", "servable": True, "shards": len(weights),
                "detail": "Fennec container"}
    packed = sum(1 for name in tensor_names(root) if name.endswith(".scales"))
    if packed:
        return {"kind": "edge0", "servable": False, "shards": len(weights), "packed": packed,
                "detail": "packed int4/int8 weights with .scales and .biases sidecars"}
    if not (root / "config.json").is_file():
        return {"kind": "unknown", "servable": False, "detail": "weights but no config.json"}
    return {"kind": "hf", "servable": True, "shards": len(weights), "detail": "HF checkpoint"}


def _means(table, names):
    """Mean of each named tensor, decoded without numpy."""
    block, final, handle, opened = [], [], None, None
    try:
        for name in names:
            entry = table.get(name)
            if not entry or entry["dtype"] not in {"F16", "BF16", "F32", "F64"}:
                continue
            if opened != entry["path"]:
                if handle:
                    handle.close()
                try:
                    handle = open(entry["path"], "rb")
                except OSError:
                    handle = None
                    break
                opened = entry["path"]
            handle.seek(entry["begin"])
            raw = handle.read(entry["end"] - entry["begin"])
            values = _decode(raw, entry["dtype"])
            if not values:
                continue
            mean = sum(values) / len(values)
            (final if name.endswith("model.norm.weight") else block).append(mean)
    finally:
        if handle:
            handle.close()
    return block, final


def _decode(raw, dtype):
    """Half, brain-float, float32 and float64 bytes as a list of floats."""
    if dtype == "F16":
        return list(struct.unpack("<%de" % (len(raw) // 2), raw[: len(raw) - len(raw) % 2]))
    if dtype == "BF16":
        words = struct.unpack("<%dH" % (len(raw) // 2), raw[: len(raw) - len(raw) % 2])
        return [struct.unpack("<f", struct.pack("<I", w << 16))[0] for w in words]
    if dtype == "F32":
        return list(struct.unpack("<%df" % (len(raw) // 4), raw[: len(raw) - len(raw) % 4]))
    if dtype == "F64":
        return list(struct.unpack("<%dd" % (len(raw) // 8), raw[: len(raw) - len(raw) % 8]))
    return None


def norm_dialect(directory) -> dict:
    """Are the stored norms gains (mean ~1) or deltas (mean ~0)?"""
    table = entries(directory) if isinstance(directory, (str, Path)) else directory
    names = [name for name in table if name.endswith(NORM_SUFFIXES)]
    block, final = _means(table, sorted(names))
    sample = sorted(block) or sorted(final)
    if not sample:
        return {"dialect": "unknown", "block_norms": 0, "median": None}
    median = sample[len(sample) // 2]
    # 0.5 is the only threshold equally wrong on both sides, and the two populations
    # sit a full unit apart, so nothing legitimate lands near it.
    return {"dialect": "gain" if median >= 0.5 else "delta", "block_norms": len(block),
            "final_norms": len(final), "median": round(median, 4)}


def target_name(name):
    """Edge0's spelling of a tensor, in the dialect the engine reads."""
    if name.startswith("language_model.model."):
        return "model.language_model." + name[len("language_model.model."):]
    # `language_model.lm_head` is also the stem of a packed matrix, whose parts are
    # spelled `.scales` and `.biases` off the bare name, so the match cannot insist on
    # the `.weight` suffix an HF name carries.
    if name == LM_HEAD or name.startswith(LM_HEAD + "."):
        return name[len("language_model."):]
    return name


def foreign_count(table) -> int:
    """Source tensors the plan leaves out because the engine never binds them."""
    return sum(1 for name in table if not name.startswith(TEXT_PREFIXES))


def decoded_shape(table, stem):
    """The float shape a packed tensor expands to: leading dims kept, columns from the
    scale count. Derived from the tensor itself, so no reference checkpoint is needed."""
    return (list(table[stem + ".weight"]["shape"][:-1])
            + [table[stem + ".scales"]["shape"][-1] * GROUP])


def packing(table, stem):
    """int4 or int8, from how many decoded columns one stored word carries."""
    weight = table[stem + ".weight"]["shape"][-1]
    groups = table[stem + ".scales"]["shape"][-1]
    if groups * GROUP == weight * 8:
        return INT4
    if groups * GROUP == weight * 4:
        return INT8
    raise ValueError("%s: %d groups x %d does not match %d packed columns"
                     % (stem, groups, GROUP, weight))


def build_plan(table) -> dict:
    """target name -> (how, source stems, output shape), the whole checkpoint planned.

    The two expert projections a layer splits across are fused into one stack here,
    which is the only structural work that cannot be done tensor by tensor: gate and up
    become the first and second halves of `mlp.experts.gate_up_proj`.
    """
    specs = {}
    done = set()
    for stem in packed_stems(table):
        if stem in done:
            continue
        shape = decoded_shape(table, stem)
        target = target_name(stem)
        if SWITCH in target:
            # The container spells an expert stack as one fused tensor with no .weight
            # suffix; every other matrix keeps the suffix the HF name has.
            target = target.replace(SWITCH, EXPERTS)
            if target.endswith(".gate_proj"):
                up = stem[: -len(".gate_proj")] + ".up_proj"
                if up + ".scales" in table and up + ".biases" in table:
                    done.update({stem, up})
                    specs[target[: -len(".gate_proj")] + ".gate_up_proj"] = (
                        "fuse", (stem, up), [shape[0], shape[1] * 2] + shape[2:])
                    continue
        else:
            target += ".weight"
        done.add(stem)
        specs[target] = ("dequant", (stem,), shape)
    for name, entry in table.items():
        part = next((tail for tail in CHECKPOINT_PARTS if name.endswith(tail)), "")
        stem = name[: -len(part)] if part else name
        base = stem[: -len(".weight")] if stem.endswith(".weight") else stem
        # A packed matrix's sidecars are folded into its decoder and a LoRA pair is
        # already merged into its weights; the matrix itself was dequantised above.
        if (part and stem + ".weight" in table) or base in done:
            continue
        if not name.startswith(TEXT_PREFIXES):
            continue
        shape = list(entry["shape"])
        # conv1d ships [C, k, 1] where the HF spelling is [C, 1, k]: the same bytes,
        # the singleton in the other slot.
        if "conv1d" in name and len(shape) == 3 and shape[1] != 1 and shape[2] == 1:
            shape = [shape[0], 1, shape[1]]
        specs[target_name(name)] = ("copy", (name,), shape)
    return specs


def plan(directory) -> dict:
    """What serving this folder would take, and whether this volume can pay for it."""
    info = classify(directory)
    out = {"kind": info["kind"], "servable": info["servable"], "detail": info.get("detail")}
    if info["kind"] == "edge0":
        table = entries(directory)
        specs = build_plan(table)
        out["action"] = "dequant"
        out["tensors"] = len(specs)
        # What the plan leaves out is part of what it reports: a checkpoint that spells
        # its language model some other way would otherwise lose tensors without a trace.
        out["foreign"] = foreign_count(table)
        out["est_bytes"] = sum(_numel(spec[2]) * 2 for spec in specs.values())
        out["dialect"] = norm_dialect(table)
        destination = output_for(directory)
        free = _free_bytes(_volume_of(destination))
        out["free_bytes"] = free
        if free is not None and free < out["est_bytes"] * 1.05:
            out["blocker"] = ("%.1f GB of converted weights needed, %.1f GB free under %s"
                              % (out["est_bytes"] / 1e9, free / 1e9, destination))
    elif info["kind"] == "hf":
        out["dialect"] = norm_dialect(directory)
        # A delta-dialect HF folder is already what the engine asks for. A gain one is
        # not, and rewriting 81 small tensors means copying the whole checkpoint to
        # change a few hundred kilobytes, so it is reported rather than done.
        out["action"] = "copy_norms" if out["dialect"]["dialect"] == "gain" else "none"
        if out["action"] == "copy_norms":
            out["blocker"] = ("norms are stored as gains but %s would be read as deltas; "
                              "the checkpoint itself disagrees with its model_type"
                              % out["dialect"].get("median"))
    return out


def _numel(shape):
    n = 1
    for dim in shape:
        n *= dim
    return n


def _volume_of(path):
    """The nearest existing ancestor, which is what a free-space query needs."""
    cursor = Path(path)
    for candidate in [cursor, *cursor.parents]:
        if candidate.exists():
            return candidate
    return Path.cwd()


def _free_bytes(path):
    try:
        return shutil.disk_usage(str(path)).free
    except OSError:
        return None


def converted_root(cache_root=None):
    """The one folder the converter owns its output in: `<cache>/converted`."""
    home = Path(cache_root or os.environ.get("FENNEC_CACHE") or (Path.home() / ".cache" / "fennec"))
    return home / "converted"


def output_for(source, cache_root=None):
    """Where a converted artifact lives: under the cache, not beside the download, so
    `scan_models` never offers the same model twice."""
    return converted_root(cache_root) / Path(source).name


def state_path(destination):
    return Path(destination) / "fennec-convert.json"


def read_state(destination):
    try:
        state = json.loads(state_path(destination).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return state if isinstance(state, dict) else None


def write_state(destination, **fields):
    """Progress and provenance, in one file the parent process is the only writer of.

    Rename rather than truncate: a reader must not catch a half-written state and
    report a running conversion as failed.
    """
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    path = state_path(destination)
    state = read_state(destination) or {}
    state.update(fields)
    state.setdefault("started_at", _now())
    state["updated_at"] = _now()
    temp = destination / "fennec-convert.json.tmp"
    temp.write_text(json.dumps(state, indent=1, sort_keys=True), encoding="utf-8")
    os.replace(temp, path)
    return state


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _bytes_of(path) -> int:
    """Everything under one folder, summed without reading a byte of it."""
    total = 0
    try:
        items = Path(path).rglob("*")
        for item in items:
            try:
                if item.is_file():
                    total += item.stat().st_size
            except OSError:
                continue
    except OSError:
        return 0
    return total


# A conversion interrupted at three in the morning leaves its shards behind and a state
# file still claiming `running`, and nothing will ever finish it. The writer stamps the
# state file once per shard, so silence for longer than a whole 35B pass means the
# writer is gone and the bytes are junk.
STALE_AFTER_S = 3 * 3600
# Formats a conversion cannot turn into something the engine reads, so a folder in one
# of them is dead weight rather than a model waiting its turn. `edge0` is deliberately
# absent: packed is exactly what this module exists to fix.
DEAD_KINDS = ("gguf", "empty", "unknown", "missing")


def _age_seconds(stamp, now):
    try:
        then = datetime.fromisoformat(str(stamp))
    except (TypeError, ValueError):
        return None
    if then.tzinfo is None:
        then = then.replace(tzinfo=timezone.utc)
    return (now - then).total_seconds()


def cleanup(models_root=None, cache_root=None, *, converted=None, models=False) -> dict:
    """Reclaim the converter's dead bytes, and price the models it cannot serve.

    Two things share the word "cleanup" here, and they differ in whose bytes they are.
    An artifact under `<cache>/converted` whose run failed or whose writer died cannot be
    served, and its source is still on disk to redo it from, so its shards are ours to
    take back. What we wrote is not always all there is to keep, though: a `failed` run
    loses only its shards, because its state file is the reason the model list says why
    this model is not servable, and a stale `running` one loses the folder, because a
    state file claiming a conversion is in progress with no writer is not information.
    A *complete* artifact whose download has since gone is a different animal: nothing
    resolves to it, but it is the last copy of that model, so it is reported as `orphaned`
    and left alone.

    A folder in the user's models root is not ours. `edge0-35b-a3b` is unsupported only
    until it is converted, and a GGUF the engine will never load is still a download
    someone chose, so `classify` findings are measured and listed and removed only when a
    caller passes `models=True`.
    """
    out_dir = Path(converted) if converted else converted_root(cache_root)
    report = {"unsupported": [], "reclaimed": [], "orphaned": [], "bytes_reclaimed": 0,
              "bytes_model": 0, "error": ""}
    root = Path(models_root) if models_root else None
    if root and root.is_dir():
        for entry in sorted(root.iterdir()):
            if not entry.is_dir() or entry.name.startswith("."):
                continue
            info = classify(entry)
            if info["kind"] not in DEAD_KINDS:
                continue
            size = _bytes_of(entry)
            report["bytes_model"] += size
            gone = {"name": entry.name, "kind": info["kind"], "bytes": size,
                    "detail": info["detail"], "removed": False}
            if models:
                try:
                    shutil.rmtree(entry)
                except OSError as exc:
                    report["error"] = "%s: %s" % (type(exc).__name__, exc)
                else:
                    gone["removed"] = True
            report["unsupported"].append(gone)
    now = datetime.now(timezone.utc)
    if not out_dir.is_dir():
        return report
    owned = out_dir.resolve()
    for entry in sorted(out_dir.iterdir()):
        if not entry.is_dir():
            continue
        state = read_state(entry)
        if state is None:
            # No state file means this folder never came from us, whatever it is.
            continue
        status = state.get("status")
        source = str(state.get("source") or "")
        if status == "complete":
            if source and not Path(source).is_dir():
                report["orphaned"].append({"name": entry.name, "source": source,
                                           "bytes": _bytes_of(entry)})
            continue
        age = _age_seconds(state.get("updated_at"), now)
        abandoned = status == "failed" or (status == "running" and age and age > STALE_AFTER_S)
        if not abandoned or not source or not Path(source).is_dir():
            continue
        target = entry.resolve()
        if owned not in target.parents:
            continue  # a link out of the cache is not a folder we may delete
        before = _bytes_of(target)
        try:
            if status == "failed":
                # Take the shards, leave the state file: it is the only carrier of the
                # reason, and `resolve` reads it to tell the model list why this model is
                # not servable. A folder deleted whole turns an explained failure back into
                # a silent "not converted yet", which is the thing that gets converted
                # again into the same ENOSPC.
                for stray in sorted(target.glob("model-*.safetensors")):
                    stray.unlink()
                scope = "shards"
            else:
                # A writer that died leaves a state file claiming `running`, and the only
                # honest repair is to remove the lie along with the bytes.
                shutil.rmtree(target)
                scope = "folder"
        except OSError as exc:
            report["error"] = "%s: %s" % (type(exc).__name__, exc)
            continue
        size = before - _bytes_of(target)
        report["reclaimed"].append({"name": entry.name, "status": status, "bytes": size,
                                    "scope": scope})
        report["bytes_reclaimed"] += size
    return report


def resolve(source, *, cache_root=None, start=True) -> dict:
    """The path to hand the engine, or why there is not one yet.

    The model list and the engine both call this. If they disagreed, the UI would offer
    a model that then serves nonsense, which is the failure this module exists to stop.
    """
    source = Path(source)
    info = classify(source)
    out = {"source": str(source), "kind": info["kind"], "path": str(source)}
    if info["kind"] not in {"edge0", "hf"}:
        out.update({"state": "ready" if info["servable"] else "unusable",
                    "detail": info.get("detail")})
        return out
    destination = output_for(source, cache_root)
    state = read_state(destination) if destination.is_dir() else None
    if state and state.get("status") == "complete":
        out.update({"state": "ready", "path": str(destination), "converted": True,
                    "provenance": {key: state.get(key) for key in
                                   ("source", "tensors", "bytes", "seconds", "dialect",
                                    "converter_version", "finished_at")}})
        return out
    if state and state.get("status") == "running":
        out.update({"state": "converting", "path": str(destination),
                    "detail": state.get("step"), "percent": state.get("percent")})
        return out
    planned = plan(source)
    if planned.get("blocker"):
        out.update({"state": "blocked", "path": str(destination), "detail": planned["blocker"],
                    "dialect": planned.get("dialect")})
        return out
    if info["kind"] == "hf":
        out.update({"state": "ready", "detail": "HF checkpoint, %s norms"
                    % planned["dialect"]["dialect"]})
        return out
    if state and state.get("status") == "failed":
        out.update({"state": "failed", "path": str(destination), "detail": state.get("error")})
        if not start:
            return out
    if not start:
        out.update({"state": "pending", "path": str(destination),
                    "detail": "conversion of %s packed tensors" % planned["tensors"]})
        return out
    if not launch(source, destination):
        out.update({"state": "blocked", "path": str(destination),
                    "detail": "conversion needs numpy: pip install fennec-core[convert]"})
        return out
    out.update({"state": "converting", "path": str(destination),
                "detail": "converting %s tensors to HF f16" % planned["tensors"]})
    return out


def launch(source, destination) -> bool:
    """Run the conversion in its own process.

    A thread dies with the server and leaves a half-written artifact that looks
    finished; a detached child outlives it, and the state file is the only channel
    back.
    """
    if not _numpy_available():
        return False
    if plan(source).get("blocker"):
        return False
    write_state(destination, status="running", step="queued", source=str(source), pid=None)
    options = ({"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt"
               else {"start_new_session": True})
    log = Path(destination) / "convert.log"
    try:
        handle = log.open("w", encoding="utf-8")
    except OSError:
        write_state(destination, status="failed", error="cannot open %s" % log)
        return False
    # The child is a fresh interpreter, so it needs this checkout on its path rather
    # than whatever the parent happened to import.
    env = dict(os.environ)
    root = str(Path(__file__).resolve().parent)
    env["PYTHONPATH"] = root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    try:
        subprocess.Popen([sys.executable, "-m", "convert", str(source),
                          str(destination)], env=env,
                         stdout=handle, stderr=handle, stdin=subprocess.DEVNULL, **options)
    except OSError as exc:
        write_state(destination, status="failed", error=str(exc))
        return False
    finally:
        handle.close()
    return True


def _numpy_available():
    try:
        import numpy  # noqa: F401
    except ImportError:
        return False
    return True


# ------------------------------------------------------------------------- conversion


def read_tensor(table, name, handles=None):
    """One source tensor as an array, reading only the bytes that belong to it.

    `handles` is the caller's {path: open file} cache: a conversion pulls 2,476 tensors
    out of 41 shards and must not reopen them for each. Without one it stays usable as
    a spot check, which is how a single decoded matrix gets compared against a
    reference artifact without writing 69 GB to find out the decode is wrong.
    """
    import numpy
    entry = table[name]
    code = {"F64": numpy.float64, "F32": numpy.float32, "I64": numpy.int64,
            "I32": numpy.int32, "I16": numpy.int16, "I8": numpy.int8, "U64": numpy.uint64,
            "U32": numpy.uint32, "U16": numpy.uint16, "U8": numpy.uint8,
            "BOOL": numpy.uint8, "F16": numpy.float16, "BF16": numpy.uint16}

    def slurp(handle):
        handle.seek(entry["begin"])
        blob = handle.read(entry["end"] - entry["begin"])
        if entry["dtype"] not in code:
            raise ValueError("%s: unsupported source dtype %s" % (name, entry["dtype"]))
        array = numpy.frombuffer(blob, dtype=code[entry["dtype"]])
        if entry["dtype"] == "BF16":
            array = (array.astype(numpy.uint32) << 16).view(numpy.float32)
        return array.reshape(entry["shape"])

    if handles is None:
        with open(entry["path"], "rb") as handle:
            return slurp(handle)
    if entry["path"] not in handles:
        handles[entry["path"]] = open(entry["path"], "rb")
    return slurp(handles[entry["path"]])


def decode(table, stem, handles=None):
    """The affine `scale * q + bias` of one packed matrix, row blocks at a time."""
    import numpy
    weight = read_tensor(table, stem + ".weight", handles)
    scales = read_tensor(table, stem + ".scales", handles)
    biases = read_tensor(table, stem + ".biases", handles)
    weight = weight.reshape(-1, table[stem + ".weight"]["shape"][-1])
    scales = scales.reshape(-1, table[stem + ".scales"]["shape"][-1])
    biases = biases.reshape(-1, table[stem + ".biases"]["shape"][-1])
    rows = weight.shape[0]
    kind = packing(table, stem)
    out = numpy.empty((rows, scales.shape[1] * GROUP), dtype=numpy.float16)
    step = 1 << 15
    for start in range(0, rows, step):
        stop = min(start + step, rows)
        block = weight[start:stop]
        if kind == INT4:
            words = block.astype(numpy.uint32)
            codes = numpy.empty((words.shape[0], words.shape[1] * 8), dtype=numpy.uint8)
            for index in range(8):
                codes[:, index::8] = (words >> (4 * index)) & 0xF
        else:
            codes = block.view(numpy.uint8).reshape(block.shape[0], block.shape[1] * 4)
        scale = numpy.repeat(scales[start:stop].astype(numpy.float32),
                             codes.shape[1] // scales.shape[1], axis=1)
        bias = numpy.repeat(biases[start:stop].astype(numpy.float32),
                            codes.shape[1] // biases.shape[1], axis=1)
        out[start:stop] = (codes.astype(numpy.float32) * scale + bias).astype(numpy.float16)
    return out.reshape(decoded_shape(table, stem))


def as_delta(array):
    """Bring a stored gain down to the delta the target's model_type implies."""
    import numpy
    flat = array.astype(numpy.float32).reshape(-1)
    if float(flat.mean()) < 0.5:
        return array.astype(numpy.float16)
    return (flat - 1.0).reshape(array.shape).astype(numpy.float16)


def convert(source, destination=None, *, quiet=True) -> dict:
    """Write the HF f16 artifact of a packed checkpoint. Blocking: call through
    `launch`, or accept that a 65 GB conversion outlives the request that asked."""
    import numpy
    began = time.time()
    source = Path(source)
    destination = Path(destination or output_for(source))
    info = classify(source)
    if info["kind"] != "edge0":
        raise ValueError("%s is %s, which has nothing to dequantise" % (source, info["kind"]))
    table = entries(source)
    specs = build_plan(table)
    dialect = norm_dialect(table)
    destination.mkdir(parents=True, exist_ok=True)
    # A previous attempt may have used a different shard count; the index is built by
    # globbing, so stragglers would be published as if they belonged to this one.
    for stale in destination.glob("model-*.safetensors"):
        stale.unlink()
    write_state(destination, status="running", step="planning", source=str(source),
                tensors=len(specs), dialect=dialect, percent=0)
    handles = {}
    order = _shards(specs)
    written = 0
    total = 0
    for index, (label, keys) in enumerate(order):
        payload = {}
        for key in keys:
            how, srcs, shape = specs[key]
            if how == "fuse":
                block = numpy.empty(shape, dtype=numpy.float16)
                cut = shape[1] // 2
                block[:, :cut] = decode(table, srcs[0], handles)
                block[:, cut:] = decode(table, srcs[1], handles)
                array = block
            elif how == "dequant":
                array = decode(table, srcs[0], handles)
            else:
                array = read_tensor(table, srcs[0], handles).astype(numpy.float16)
                if list(array.shape) != shape:
                    array = array.reshape(shape)
            if key.endswith(NORM_SUFFIXES):
                array = as_delta(array)
            payload[key] = numpy.ascontiguousarray(array)
            written += 1
            del array
        name = "model-%05d-of-%05d.safetensors" % (index + 1, len(order))
        total += _write_safetensors(destination / name, payload)
        del payload
        write_state(destination, status="running", tensors_done=written, tensors=len(specs),
                    percent=round(100 * (index + 1) / len(order)),
                    step="%s (%d/%d shards)" % (label, index + 1, len(order)))
        if not quiet:
            print("%s %s %d/%d" % (name, label, index + 1, len(order)), flush=True)
    for handle in handles.values():
        handle.close()
    _sidecars(source, destination)
    _index(destination)
    state = write_state(destination, status="complete", step="done", percent=100,
                        tensors=written, bytes=total, source=str(source), dialect=dialect,
                        converter_version=VERSION, seconds=round(time.time() - began, 1),
                        finished_at=_now())
    # Finishing is the moment to reclaim the runs that did not: it is the one time this
    # process knows the volume was tight enough to convert into, and a failed sibling
    # costs the next conversion its free space.
    swept = cleanup(converted=Path(destination).parent)
    if swept["reclaimed"]:
        state = write_state(destination, reclaimed=swept["reclaimed"],
                            reclaimed_bytes=swept["bytes_reclaimed"])
    return state


def _shards(specs):
    """(label, keys) groups: globals first, the head apart from them, then one shard
    per layer.

    Keeps the embedding and lm_head out of memory at the same time as anything else,
    and makes an interrupted run resume by rewriting whole files.
    """
    flat = sorted(specs)
    globals_ = [key for key in flat if ".layers." not in key]
    head = [key for key in globals_ if key == "lm_head.weight"]
    body = [key for key in globals_ if key != "lm_head.weight"]
    order = ([("globals", body)] if body else [])
    if head:
        order.append(("lm_head", head))
    layers = sorted({int(key.split(".layers.")[1].split(".")[0]) for key in flat
                     if ".layers." in key})
    for layer in layers:
        order.append(("layer %d" % layer,
                      [key for key in flat if ".layers.%d." % layer in key]))
    return [(label, [key for key in keys if key in specs]) for label, keys in order]


def _write_safetensors(path, payload):
    """name -> contiguous array as a .safetensors file. Apart from the arrays
    themselves this is stdlib: a length, a JSON table, then the bytes."""
    codes = {("float", "16"): "F16", ("float", "32"): "F32", ("float", "64"): "F64",
             ("int", "8"): "I8", ("int", "16"): "I16", ("int", "32"): "I32",
             ("int", "64"): "I64", ("uint", "8"): "U8", ("uint", "16"): "U16",
             ("uint", "32"): "U32", ("uint", "64"): "U64", ("bool", "8"): "BOOL"}
    table, offset = {}, 0
    for name, array in payload.items():
        kind = array.dtype.kind
        size = str(array.dtype.itemsize * 8)
        code = codes.get((("bool" if kind == "b" else
                           {"u": "uint", "i": "int", "f": "float"}.get(kind, kind)), size))
        if code is None:
            raise ValueError("%s: cannot write dtype %s" % (name, array.dtype))
        table[name] = {"dtype": code, "shape": list(array.shape),
                       "data_offsets": [offset, offset + int(array.nbytes)]}
        offset += int(array.nbytes)
    blob = json.dumps(table, separators=(",", ":")).encode("utf-8")
    # The data section starts right after the header, so the header is space-padded to
    # keep every tensor's bytes 8-aligned.
    blob += b" " * ((-len(blob)) % 8)
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(blob)))
        handle.write(blob)
        for array in payload.values():
            handle.write(array.tobytes())
    return 8 + len(blob) + offset


def _index(destination):
    table, total = {}, 0
    for shard in sorted(Path(destination).glob("model-*.safetensors")):
        header = safetensors_header(shard)
        total += shard.stat().st_size
        for name in header:
            table[name] = shard.name
    body = {"metadata": {"total_size": total}, "weight_map": table}
    (Path(destination) / "model.safetensors.index.json").write_text(
        json.dumps(body, indent=1), encoding="utf-8")
    return table


def _sidecars(source, destination):
    """The tokenizer and config belong to the checkpoint; the converter adds no opinion
    about chat templates or sampling."""
    missing = []
    for name in SIDECARS:
        candidate = Path(source) / name
        if candidate.is_file():
            shutil.copy2(candidate, Path(destination) / name)
        elif name in {"config.json", "tokenizer.json"}:
            missing.append(name)
    if missing:
        raise ValueError("%s: source has no %s, so the converted folder could not be read"
                         % (source, " or ".join(missing)))


def _human(size) -> str:
    """Big enough to be worth deleting in the units a deletion decision is made in."""
    return ("%.1f GiB" % (size / 2**30)) if size >= 2**30 else ("%.0f MiB" % (size / 2**20))


USAGE = ("usage: fennec-convert "
         "[--inspect DIR | --cleanup [MODELS_ROOT] [--models] | SOURCE [DESTINATION]]")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in ("-h", "--help"):
        print(USAGE)
        return 0
    if argv and argv[0] == "--inspect":
        if len(argv) < 2:
            print("usage: python -m convert --inspect MODEL_DIR", file=sys.stderr)
            return 2
        directory = argv[1]
        print(json.dumps({"classify": classify(directory), "plan": plan(directory)}, indent=1))
        return 0
    if argv and argv[0] == "--cleanup":
        rest = [item for item in argv[1:] if not item.startswith("--")]
        report = cleanup(rest[0] if rest else None, models="--models" in argv)
        print(json.dumps(report, indent=1))
        for line in (["reclaimed %s of %s: %s" % (item["scope"], item["name"],
                                                  _human(item["bytes"]))
                      for item in report["reclaimed"]]
                     + ["unsupported %s: %s (%s)"
                        % (item["name"], _human(item["bytes"]), item["detail"])
                        for item in report["unsupported"]]
                     + ["orphaned %s: %s, kept (its source is gone)"
                        % (item["name"], _human(item["bytes"]))
                        for item in report["orphaned"]]
                     + (["error: %s" % report["error"]] if report["error"] else [])):
            print(line, file=sys.stderr, flush=True)
        return 0
    if not argv:
        print(USAGE, file=sys.stderr)
        return 2
    if argv[0].startswith("-"):
        print("%s\nfennec-convert: unknown option %s" % (USAGE, argv[0]), file=sys.stderr)
        return 2
    source = Path(argv[0])
    destination = Path(argv[1]) if len(argv) > 1 else output_for(source)
    try:
        state = convert(source, destination, quiet=False)
    except Exception as exc:
        # The state file is the only thing the parent will ever read back.
        write_state(destination, status="failed", step="error", source=str(source),
                    error="%s: %s" % (type(exc).__name__, exc))
        print("conversion failed: %s" % exc, file=sys.stderr, flush=True)
        return 1
    print("converted %s -> %s (%s tensors, %.1f GB, %.0fs)"
          % (source, destination, state["tensors"], state["bytes"] / 1e9, state["seconds"]),
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
