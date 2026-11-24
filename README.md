![Fennec Core 1.0.0 — the shared runtime behind every Fennec app: engine supervisor, bounded agent loop, workspace-jailed tools, ACP v1 for editors, and a FastAPI service hosting the browser workspace. Python 3.10+, Rust edition 2024, 535 Python tests, 6 Rust crates, 66 Rust tests, MIT license.](docs/hero.png)

---

## At a glance

| Python tests | Rust tests | Rust crates | Python modules | Runtime deps | `fennec-rs` binary |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 535 in 14 files | 66 in 6 crates | 6 | 16 | 4 | ≈ 1.9 MB |

Counts measured on 2026-11-24 from `tests/` (535 collected, 1 skipped) and `src/` (6 crates). Binary size measured from
`fennec-cli/target/release/fennec-rs.exe` (Windows x64 `cargo build --release --offline`, 1,897,180 bytes at 1.0.0).

One reference implementation for the desktop, coding-agent and JetBrains apps. Pure logic is
also ported to a six-crate Rust workspace that the Python side can delegate to.

## Architecture

The supported coding-agent browser entry point is the separate Rust/Axum
[`fennec-agent`](https://github.com/Liso7150/fennec-coding-agent), which serves the PWA and
its API on `127.0.0.1:8787` and falls back to `127.0.0.1:8788` when that port is occupied.
This repository remains the Python orchestration boundary used by the agent loop, ACP,
engine supervision, and optional connectors:

```
 Rust/Axum fennec-agent               Editors (ACP v1, stdio)
        │                                 │
        └───────────────┬─────────────────┘
                        ▼
          Python fennec-core when selected
                        │
                        ▼
   loop.py ── bounded agent loop, step cap, write/shell approvals
        │
        ├── tools.py ──── read / edit / grep / run / git inside a workspace jail
        ├── context.py ── payload compaction (the UI transcript is never cut)
        ├── outlets.py ── Fennec's d1.2 loop / OpenBot, gated + redacted, off unless configured
        └── engine.py ── attach-first supervisor
                 │
                 ▼
        Fennec native binary on 127.0.0.1:8184, spawned on demand
```

It stops the child it owns with all descendants, never a server it attached to. A built
[Fennec](https://github.com/Liso7150/fennec) binary wins: `target/release` → `target/debug` → root.

## Install

Requires Python 3.10+.

```bash
python -m pip install -e .            # core
python -m pip install -e ".[stt]"     # + local speech transcription
python -m pip install -e ".[antigravity]" # + Google Antigravity outlet
python -m pip install -e ".[dev]"     # + httpx for the test client
```

| Extra | Pins | Why |
|---|---|---|
| runtime | `fastapi`, `uvicorn`, `psutil`, `pydantic` | service, process supervision, validation |
| `stt` | `faster-whisper==1.2.1`, `av>=14,<18` | PyAV 18+ broke faster-whisper 1.2.1's `metadata_errors` |
| `antigravity` | `google-antigravity>=0.1.24` | optional Google agent outlet |
| `dev` | `httpx>=0.27` | test client |

The Antigravity outlet is off unless `google-antigravity` is installed, `GEMINI_API_KEY`
(or Vertex AI environment variables) is configured, and `FENNEC_WORKSPACE` points to an
existing directory. It exposes only Fennec's jailed `list_dir`, `read_file`, and `grep`
tools; Antigravity's built-in shell and filesystem tools are denied.

## Quick start

For the supported coding-agent PWA, build and run the Rust/Axum server from
[`fennec-coding-agent`](https://github.com/Liso7150/fennec-coding-agent):

```bash
cargo build --release --offline
./target/release/fennec-agent
```

It serves the PWA and API on `127.0.0.1:8787`, falling back to `127.0.0.1:8788` when needed.
The Python core service remains available when a core-specific route or legacy integration
requires it:

```bash
python -m fennec_core.server serve --host 127.0.0.1 --port <free-port>
```

The engine listens on `127.0.0.1:8184`, started only when a message needs it.

> [!IMPORTANT]
> A non-loopback bind requires an app token (`FENNEC_APP_TOKEN`) **and** TLS in front of the
> service. Host and Origin are checked on every request; plain LAN HTTP is refused.

### Connect Nod (n8n-compatible)

The native `fennec-nod` stdio binary lives in `fennec-coding-agent`. It exposes only Fennec status,
jailed workspace listing, jailed file reading, jailed grep, and direct local-model chat. It does not
expose shell, Python, file writes, browser automation, coworker delegation, or the bounded agent
loop. Nod remains a separate application and connects through its normal MCP connector flow.

### Obsidian boundary

Obsidian support lives in the Rust/Axum `fennec-agent` backend as an opt-in local-vault writer. The
vault is selected with `FENNEC_OBSIDIAN_VAULT` or `OBSIDIAN_VAULT`; titles and folders are jailed,
existing notes are protected unless overwrite is explicit, and Obsidian remains responsible for
account login and Sync. Fennec does not embed Obsidian, Node.js, Electron, or a plugin runtime.

### Subcommands

| Command | What it does |
|---|---|
| `fennec-agent serve` | HTTP API + PWA host (`--host`, `--port`, `--web-root`) |
| `fennec-agent status --url <base>` | engine health and model id |
| `fennec-agent chat <prompt>` | one-shot chat against the engine |
| `fennec-agent agent <task>` | bounded tool loop (`--allow-write`, `--allow-shell`, `--allow-web`, `--max-steps`) |
| `fennec-agent acp` | speak ACP v1 on stdio; `--check` reports what is advertised |
| `fennec-agent outlets` | status of the optional outlets: Fennec's own d1.2 loop, OpenBot, Antigravity |

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
| `outlets.py` | outlet gates (`web_task`, `coworker_task`, and optional `antigravity_task`), interpreter and transport discovery, workspace jail, host allow-list, secret redaction |
| `mcp.py` | MCP server pool |
| `preflight.py` | weights + peak KV cache against this machine's memory |
| `server.py` | Separate Python core service, routes, loopback guards, and legacy/static integrations |
| `stt.py` | faster-whisper transcription sidecar on `127.0.0.1:8183` |
| `rust_bridge.py` | optional delegation to the Rust crates |
| `config.py` | the shared config contract: `~/.fennec/config.json` with `FENNEC_*` overrides |
| `setup_wizard.py` | headless first-run setup behind `fennec setup` |
| `webagent.py` | Fennec's own browser loop: the page-reader JS in `snapshot.py` offers indices, the d1 model picks one, this resolves and drives it over CDP |
| `web_driver.py` | that loop's process boundary: launched with the interpreter that has the browser transport |
| `liquid.py` | Liquid decision client: hosted `d1.2` (`LIQUID_API_KEY`) or a free open-weight `d1-2B` on loopback (`LIQUID_ENDPOINT`); stdlib HTTP(S) only |
| `te_bridge.py` | external tokenizer/economy bridge |

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
| `fennec-acp` | ACP frame encoding, clipping, piece splitting |
| `fennec-outlets` | secret redaction and blocked-host checks |
| `fennec-preflight` | the memory preflight plan |
| `fennec-client` | the loopback protocol clients (agent server + engine SSE) and suite/manifest discovery |
| `fennec-workspace-tools` | the workspace-jailed read/edit/grep/run tools and the jail path that must match the PWA byte for byte |

Context compaction is not here: it has one Rust implementation, in
[fennec-economy](https://github.com/Liso7150/fennec-economy), which `fennec-core`, the
desktop app and the CLI all path-depend on. An earlier copy of the crate under `src/` was
deleted once nothing consumed it.

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

**Measured 2026-11-10**, on a representative transcript against a local Colibri-served
Qwen3.6-35B-A3B endpoint (`qwen3.6-colibri` on `127.0.0.1:8197`), 2 order-counterbalanced
pairs, `max_tokens=8`, `temperature=0`:

| Arm | Payload chars | Prompt tokens | Median TTFT |
|---|---|---|---|
| old loop | 25,068 | 9,594 | 70.703 s |
| compacted | 19,066 | 6,995 | 51.141 s |
| **delta** | **−23.9 %** | **−2,599** | **−27.7 %** |

Compaction itself reduced a 52,543-character transcript to 19,413 (3 results elided, 1
deduplicated). Source: `fennec-finetune/results/context_payload.json`, produced by
`fennec-finetune/scripts/bench_context_payload.py`. Two pairs is a small sample: the TTFT
delta is a measurement on this host, not a general latency claim, and it is **not a Fennec
figure** — the endpoint that served it was Colibri.

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
> `--engine-root` must be a Fennec checkout carrying a built binary (`fennec` under
> `target/release`, `target/debug` or the root). Anything else — an unbuilt checkout,
> a foreign tree, a missing path — fails validation with the same remedy: build the
> engine with `cargo build --release` or point `engine_root` elsewhere.

## Tests

```bash
python -m unittest discover -s tests -v     # 535 tests across 14 files, 1 skipped
cd src && cargo test --release --offline    # 66 tests across 16 suites
```

Everything is hermetic: no engine, no network, no model. The tokenizer/economy bridge is
forced off (`FENNEC_TOKENECONOMIZER=0`), outlet HTTP goes through a mocked `build_opener`,
the d1.2 decision client takes an injected fake opener (so no call is ever paid for),
and outlet subprocesses are mocked too. Rust goldens pin the exact byte formatting the
Python side produces, so the two implementations cannot silently drift.

## Credits

Engine: [Liso7150/fennec](https://github.com/Liso7150/fennec). Base idea:
[JustVugg/colibri](https://github.com/JustVugg/colibri) — device-tiered MoE inference,
replaced by Fennec and never launched here. Fine-tuning: [AirLLM](https://github.com/lyogavin/airllm).
Browser policy: a Liquid **d1** decision model, used only through `liquid.py` - the
hosted `d1.2`
(`https://api.liquid.ai/decisions/v1/systemone`, paid, `LIQUID_API_KEY`) or the
free open-weight `d1-2B` on loopback (`LIQUID_ENDPOINT`, no key, $0.00). The loopback
half is not provisioned here: `d1-2B` speaks the same System One envelope, and Fennec's
own engine serves `/v1/chat/completions`, not `/v1/systemone`, so that lane needs a
System One-compatible model server (llama.cpp >= 0.8.0) already answering the URL you
point `LIQUID_ENDPOINT` at. The page reader `snapshot.py`,
which holds that JS as its `JS` constant, is derived from `snapshot.js` in [browser-use/jev-ultrafast](https://github.com/browser-use/jev-ultrafast)
(MIT License, Copyright (c) 2026 Browser Use) — attribution for derivative code, not a
dependency: that checkout drives nothing here. The lane's CDP transport is
[browser-harness](https://github.com/browser-use/browser-harness), installed from PyPI into
Fennec's own browser venv (`~/.fennec/browser-tools/fennec/.venv`) instead of the interpreter
the apps run on.
Part of the [Fennec suite](https://github.com/Liso7150/fennec-environment-setup).