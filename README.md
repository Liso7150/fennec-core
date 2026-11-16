![Fennec Core 2.0.0 — the shared runtime behind every Fennec app: engine supervisor, bounded agent loop, workspace-jailed tools, ACP v1 for editors, and a FastAPI service hosting the browser workspace. Python 3.10+, Rust edition 2024, 251 Python tests, 6 Rust crates, 53 Rust tests, MIT license.](docs/hero.png)

---

## At a glance

| Python tests | Rust tests | Rust crates | Python modules | Runtime deps | `fennec-rs` binary |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 251 in 10 files | 53 in 6 crates | 6 | 16 | 4 | ≈ 1.9 MB |

Counts measured on 2026-11-15 from `tests/` (251 collected, 1 skipped) and `src/`. Binary size measured from
`fennec-cli/target/release/fennec-rs.exe` (Windows x64 `cargo build --release --offline`, 1,897,180 bytes at 1.0.0).

One reference implementation for the desktop, coding-agent and JetBrains apps. Pure logic is
also ported to a six-crate Rust workspace that the Python side can delegate to.

## Architecture

```
 PWA / HTTP clients               Editors (ACP v1, stdio)
        │                                 │
        ▼                                 ▼
   server.py ◄─────────────────────── acp.py
   FastAPI + loopback guards              │
        │                                 │
        ▼                                 ▼
   loop.py ── bounded agent loop, step cap, write/shell approvals
        │
        ├── tools.py ──── read / edit / grep / run / git inside a workspace jail
        ├── context.py ── payload compaction (the UI transcript is never cut)
        ├── outlets.py ── Jev / OpenBot, gated + redacted, off unless configured
        └── engine.py ── attach-first supervisor
                 │
                 ▼
        Fennec native binary on 127.0.0.1:8184, spawned on demand
        (legacy Colibri c/coli under Python as fallback)
```

It stops the child it owns with all descendants, never a server it attached to. A built
[Fennec](https://github.com/Liso7150/fennec) binary wins: `target/release` → `target/debug` → root.

## Install

Requires Python 3.10+.

```bash
python -m pip install -e .            # core
python -m pip install -e ".[stt]"     # + local speech transcription
python -m pip install -e ".[dev]"     # + httpx for the test client
```

| Extra | Pins | Why |
|---|---|---|
| runtime | `fastapi`, `uvicorn`, `psutil`, `pydantic` | service, process supervision, validation |
| `stt` | `faster-whisper==1.2.1`, `av>=14,<18` | PyAV 18+ broke faster-whisper 1.2.1's `metadata_errors` |
| `dev` | `httpx>=0.27` | test client |

## Quick start

```bash
fennec-agent serve --host 127.0.0.1 --port 8787
```

Serves the HTTP API and mounts the [coding-agent PWA](https://github.com/Liso7150/fennec-coding-agent)
at `/`. The engine listens on `127.0.0.1:8184`, started only when a message needs it.

> [!IMPORTANT]
> A non-loopback bind requires an app token (`FENNEC_APP_TOKEN`) **and** TLS in front of the
> service. Host and Origin are checked on every request; plain LAN HTTP is refused.

### Subcommands

| Command | What it does |
|---|---|
| `fennec-agent serve` | HTTP API + PWA host (`--host`, `--port`, `--web-root`) |
| `fennec-agent status --url <base>` | engine health and model id |
| `fennec-agent chat <prompt>` | one-shot chat against the engine |
| `fennec-agent agent <task>` | bounded tool loop (`--allow-write`, `--allow-shell`, `--allow-web`, `--max-steps`) |
| `fennec-agent acp` | speak ACP v1 on stdio; `--check` reports what is advertised |
| `fennec-agent outlets` | status of the optional Jev / OpenBot outlets |

Reasoning pass-through (`reasoning_effort` / `enable_thinking`, `reasoning_content` deltas)
and one validated `data:image/...;base64` vision part per request are supported. Tools are
read-only by default; the speech sidecar (faster-whisper, CPU) sits behind `POST /api/stt`.

## Modules

| Module | Responsibility |
|---|---|
| `engine.py` | engine lifecycle, health, model-id discovery, telemetry (`/health` forwarded verbatim, so an engine-declared capability like `vision` reaches the shells without a new core field; plus `/metrics`, `/experts`, `/profile`), embeddings, STT lifecycle |
| `acp.py` | ACP v1 `Agent` and `Session`, stdio framing |
| `loop.py` | bounded agent loop: tool-call accumulation, result feedback, step cap |
| `tools.py` | read / edit / grep / run / git inside a workspace jail, plus optional NPU semantic search |
| `context.py` | payload compaction: head+tail elision, dedup, total budget |
| `outlets.py` | Jev / OpenBot clients, host allow-list, secret redaction |
| `mcp.py` | MCP server pool |
| `preflight.py` | weights + peak KV cache against this machine's memory |
| `server.py` | FastAPI app, routes, static PWA mount, loopback guards |
| `stt.py` | faster-whisper transcription sidecar on `127.0.0.1:8183` |
| `rust_bridge.py` | optional delegation to the Rust crates |
| `config.py` | the shared config contract: `~/.fennec/config.json` with `FENNEC_*` overrides |
| `setup_wizard.py` | headless first-run setup behind `fennec setup` |
| `jev_driver.py` | browser-outlet driver |
| `tokeneconomizer.py` | external tokenizer/economy bridge |

> [!NOTE]
> NPU semantic search is **off by default**. It needs an embedding endpoint and is enabled
> explicitly; nothing probes for an NPU unless you ask.

## Rust workspace

`src/` holds six crates, all at version 1.0.0, edition 2024. They port pure logic —
no inference, no model weights. Each is the only implementation of its feature: the desktop
app, the JetBrains sidecar and the CLI path-depend on these, they do not copy them.

| Crate | What it ports |
|---|---|
| `fennec-pyjson` | byte-exact `json.dumps` formatting, so Rust and Python agree on the wire |
| `fennec-economy` | context compaction (`compact`) |
| `fennec-acp` | ACP frame encoding, clipping, piece splitting |
| `fennec-outlets` | secret redaction and blocked-host checks |
| `fennec-preflight` | the memory preflight plan |
| `fennec-client` | the loopback protocol clients (agent server + engine SSE) and suite/manifest discovery |

`fennec-rs`, the CLI binary over the same crates, lives in
[fennec-cli](https://github.com/Liso7150/fennec-cli).

Delegation is opt-in via `FENNEC_RUST=1` and only above a payload-size gate
(`FENNEC_RUST_MIN_CHARS`), so the Python path stays the reference.

```bash
cargo build --release --offline
cargo test --offline
```

## Context economy

The loop compacts only what it sends to the model: a tool result keeps its head and tail, a
repeated result becomes `[identical to the result of <call>; N chars not repeated]`, and a
total budget shrinks the oldest tool results first — never the system prompt, the user task,
or the newest result.

| Variable | Default | Meaning |
|---|---|---|
| `FENNEC_AGENT_RESULT_CHARS` | 6000 | per-result cap (floor 600) |
| `FENNEC_AGENT_CONTEXT_BUDGET` | 48000 | characters sent per step (never below the result cap) |

Savings are reported per session in `/api/agent/poll` usage as `context_chars_before`,
`context_chars_after`, `context_elided` and `context_deduped`.

**Measured**, on a representative transcript against a local Colibri-served Qwen3.6-35B-A3B
endpoint, 2 order-counterbalanced pairs, `max_tokens=8`, `temperature=0`:

| Arm | Payload chars | Prompt tokens | Median TTFT |
|---|---|---|---|
| old loop | 25,068 | 9,594 | 70.703 s |
| compacted | 19,066 | 6,995 | 51.141 s |
| **delta** | **−23.9 %** | **−2,599** | **−27.7 %** |

Compaction itself reduced a 52,543-character transcript to 19,413 (3 results elided, 1
deduplicated). Source: `fennec-finetune/results/context_payload.json`, produced by
`fennec-finetune/scripts/bench_context_payload.py`. Two pairs is a small sample: the TTFT
delta is a measurement on this host, not a general latency claim.

Engine benchmarks: see `benchmarks/` in the suite root.

## Headless setup

`fennec setup` (from [fennec-cli](https://github.com/Liso7150/fennec-cli)) runs
`setup_wizard.py` on a server or CI box: detects Python, engine and models, checks deps, and
writes the desktop wizard's keys — never a secret: `engine_root`, `suite_root`, `models_root`,
`cache_root`, `python`, `model`, `engine_url`, `web_root`, `workspace`, `backend`, `ctx`,
`policy`. An existing config is backed up to `config.json.bak-fennec-setup`, replaced atomically.

```bash
fennec setup                                       # prompts with detected defaults
fennec setup --yes --engine-root /opt/fennec       # noninteractive
fennec setup --dry-run --json                      # print the config, write nothing
fennec setup --yes --install-deps                  # pip install missing core deps first
```

`backend` is one of `auto`, `cuda`, `vulkan`, `metal`, `cpu`; `policy` one of `quality`,
`balanced`, `experimental-fast`. Models are optional.

> [!NOTE]
> `--engine-root` accepts either a native Fennec checkout (a `fennec` binary under
> `target/release`, `target/debug` or the root) or a Colibri checkout carrying
> `c/coli` and `c/openai_server.py`. The engine runtime prefers the native binary;
> `c/coli` is the legacy fallback.

## Tests

```bash
python -m unittest discover -s tests -v     # 251 tests across 10 files, 1 skipped
cd src && cargo test --offline             # 53 tests across the 6 crates
```

Everything is hermetic: no engine, no network, no model. The tokenizer/economy bridge is
forced off (`FENNEC_TOKENECONOMIZER=0`), outlet HTTP goes through a mocked `build_opener`,
and outlet subprocesses are mocked too. Rust goldens pin the exact byte formatting the
Python side produces, so the two implementations cannot silently drift.

## Credits

Engine: [Liso7150/fennec](https://github.com/Liso7150/fennec). Base idea:
[JustVugg/colibri](https://github.com/JustVugg/colibri) — device-tiered MoE inference, still
supported as the legacy fallback. Fine-tuning: [AirLLM](https://github.com/lyogavin/airllm).
Part of the [Fennec suite](https://github.com/Liso7150/fennec-environment-setup).