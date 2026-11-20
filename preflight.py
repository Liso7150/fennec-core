"""Device-aware model preflight: weights plus peak KV cache against this machine.

The spec-sheet number that matters is not the model's parameter count but the
memory a run actually needs: the weights, plus the KV cache for the context the
session will really use. The engine tiers weights across VRAM, RAM and SSD, but
the KV cache has to stay in memory, so it is checked separately.

Everything is read-only and offline: no model is loaded and no request is made.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

# BF16/FP16 key and value tensors; the engine keeps the cache at the model dtype.
BYTES_PER_ELEMENT = 2


def _dir_bytes(path: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total


def _text_config(data: dict) -> dict:
    nested = data.get("text_config")
    return nested if isinstance(nested, dict) else data


def kv_bytes_per_token(model_dir) -> int | None:
    """KV cache growth per token, from the model's own config, or None."""
    try:
        data = json.loads((Path(model_dir) / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    cfg = _text_config(data)
    layers = cfg.get("num_hidden_layers")
    heads = cfg.get("num_key_value_heads") or cfg.get("num_attention_heads")
    head_dim = cfg.get("head_dim")
    if not head_dim:
        hidden, attention = cfg.get("hidden_size"), cfg.get("num_attention_heads")
        if isinstance(hidden, int) and isinstance(attention, int) and attention > 0:
            head_dim = hidden // attention
    try:
        layers, heads, head_dim = int(layers), int(heads), int(head_dim)
    except (TypeError, ValueError):
        return None
    elements = 2 * layers * heads * head_dim          # one key and one value per layer
    return elements * BYTES_PER_ELEMENT if elements > 0 else None


def _nvidia_vram() -> int | None:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        done = subprocess.run([exe, "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    if done.returncode != 0:
        return None
    try:
        return int(done.stdout.strip().splitlines()[0]) * 1024 * 1024
    except (ValueError, IndexError):
        return None


def _ram_available() -> int | None:
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except Exception:
        return None


def disk_free(path) -> int | None:
    try:
        return int(shutil.disk_usage(str(path)).free)
    except OSError:
        return None


def plan(model_dir, ctx: int = 8192, *, kv_bytes: int | None = None,
         vram_bytes: int | None = None, ram_available_bytes: int | None = None) -> dict:
    """What this model needs for a `ctx`-token session, and whether it fits."""
    path = Path(model_dir).expanduser()
    weights = _dir_bytes(path) if path.is_dir() else 0
    per_token = kv_bytes if kv_bytes is not None else kv_bytes_per_token(path)
    kv = per_token * ctx if per_token else None
    vram = vram_bytes if vram_bytes is not None else _nvidia_vram()
    ram = ram_available_bytes if ram_available_bytes is not None else _ram_available()
    notes: list[str] = []
    if weights <= 0:
        notes.append("model directory is missing or empty")
    if per_token is None:
        notes.append("KV size unknown: config.json has no usable layer/head geometry")
    if vram is None:
        notes.append("no NVIDIA GPU reported; treating VRAM as zero")

    budget = (vram or 0) + (ram or 0)
    if weights <= 0:
        fit = "unknown"
    elif vram and weights <= vram:
        fit = "vram"
    elif budget and weights <= budget:
        fit = "hybrid"
    else:
        fit = "disk"
    kv_fits = None if kv is None else bool(budget) and kv <= budget
    if kv_fits is False:
        notes.append("KV cache alone exceeds VRAM plus available RAM at this context")
    free = disk_free(path if path.is_dir() else path.parent)
    return {
        "model": str(path),
        "ctx": ctx,
        "weights_bytes": weights,
        "kv_bytes_per_token": per_token,
        "kv_bytes": kv,
        "peak_bytes": weights + kv if kv is not None else None,
        "vram_bytes": vram,
        "ram_available_bytes": ram,
        "disk_free_bytes": free,
        "weights_fit": fit,
        "kv_fits": kv_fits,
        "notes": notes,
    }


def human(report: dict) -> list[str]:
    """One line per fact, for a terminal."""
    def gb(value):
        return "unknown" if value is None else f"{value / 1024 ** 3:.2f} GiB"

    lines = [
        f"model        {report['model']}",
        f"weights      {gb(report['weights_bytes'])}",
        f"context      {report['ctx']} tokens",
        f"KV per token {gb(report['kv_bytes_per_token']) if report['kv_bytes_per_token'] else 'unknown'}",
        f"KV at ctx    {gb(report['kv_bytes'])}",
        f"peak         {gb(report['peak_bytes'])}",
        f"VRAM         {gb(report['vram_bytes'])}",
        f"RAM free     {gb(report['ram_available_bytes'])}",
        f"weights fit  {report['weights_fit']}",
        f"KV fits      {report['kv_fits']}",
    ]
    lines.extend(f"note         {note}" for note in report["notes"])
    return lines
