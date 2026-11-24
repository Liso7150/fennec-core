![Fennec Core 1.0.0 — the shared runtime behind every Fennec app: engine supervisor, bounded agent loop, workspace-jailed tools, ACP v1 for editors, and a FastAPI service hosting the browser workspace. Python 3.10+, Rust edition 2024, 561 Python tests, 6 Rust crates, 69 Rust tests, MIT license.](docs/hero.png)

---

## At a glance

| Python tests | Rust tests | Rust crates | Python modules |
|:---:|:---:|:---:|:---:|
| 560 passed, 1 skipped in 15 files | 69 in 6 crates | 6 | 23 |

The Python modules are top-level modules rather than a `fennec_core` package. Run the service
from this repository with `python server.py serve`.

## Architecture

`fennec-core/server.py` is the canonical supervised coding backend. It serves the browser PWA
and API on `127.0.0.1:8787` and advertises the capabilities clients use to select a coding route:

- `supervised-agent`
- `tool-execution`
- `per-call-approvals`
- `agent-polling`

It also advertises `workspace-jail`, `workspace-write`, `shell-unsandboxed`, and `pwa-host`.
The PWA files live in `fennec-coding-agent/web` by default; core serves them at its own origin.

```text
Browser PWA / native desktop / ACP editors
                    │
                    ▼
       fennec-core/server.py :8787
       supervised loop + API + PWA host
                    │
      ┌─────────────┼────────────────┐
      ▼             ▼                ▼
 tools.py       engine.py         outlets.py
 workspace      supervised       optional,
 jail + caps     Fennec process  gated
                    │
                    ▼
        native Fennec engine :8184
```

The separate Rust/Axum `fennec-agent` service is auxiliary chat and integration infrastructure
on exactly `127.0.0.1:8788`, overridable with `FENNEC_AGENT_ADDR`. It advertises `chat-only`
and is not a coding fallback. The engine supplies inference and telemetry on
`127.0.0.1:8184`; core owns supervised tool execution and approvals.

Core stops an engine process it spawned, including descendants. It never stops an engine it
merely attached to. An existing built Fennec binary is discovered under `target/release`, then
`target/debug`, then the engine checkout root.

## Install

Requires Python 3.10+.

```bash
python -m pip install -e .                 # runtime
python -m pip install -e ".[models]"       # model downloads
python -m pip install -e ".[dev]"          # test client
python -m pip install -e ".[stt]"          # local speech transcription
python -m pip install -e ".[antigravity]"  # optional outlet
python -m pip install -e ".[convert]"      # converted-tensor materialization
```

| Extra | Purpose |
|---|---|
| runtime | FastAPI, Uvicorn, psutil, and Pydantic |
| `models` | `huggingface_hub>=0.25` for the recommended model download |
| `dev` | `httpx>=0.27` for the test client |
| `stt` | faster-whisper 1.2.1 with `av>=14,<18` |
| `convert` | NumPy for materializing converted tensors |
| `antigravity` | Google Antigravity outlet client |

The Antigravity outlet is disabled unless its package and credentials are configured and
`FENNEC_WORKSPACE` names an existing directory. It exposes only jailed `list_dir`, `read_file`,
and `grep`; its own shell and filesystem tools are denied.

## Quick start

Build the Fennec engine first, then install core and provision a model:

```bash
python -m pip install -e ".[models]"
fennec setup --yes --download-model --engine-root <fennec-checkout>
python server.py serve
```

Open `http://127.0.0.1:8787`. The service starts or attaches to the engine when a request needs
inference.

### Model setup

`--download-model` downloads the Fennec-ready
[`Liso/qwen36-35b-colibri`](https://huggingface.co/Liso/qwen36-35b-colibri) checkpoint into
`<models-root>/qwen36-35b` and selects it. The download is up to **22.78 GiB**, keeps existing
files, can resume, and verifies the required JSON, tokenizer, and safetensors files. Set
`HF_TOKEN` in the environment when the repository requires authentication; setup does not save
the token.

An existing model can be selected with `fennec setup --model <directory>` or `FENNEC_MODEL`.
A relative model name resolves under `models_root`. The supported serving path is native
Fennec/Hugging Face safetensors or a Fennec container. GGUF is classified as a llama.cpp
container and is not directly servable by Fennec. Packed Edge0 weights with `.scales` or
`.biases` require conversion before serving.

> [!IMPORTANT]
> A non-loopback service bind requires `FENNEC_APP_TOKEN` and a TLS certificate/key. Host and
> Origin are checked on requests; plain LAN HTTP is refused. API requests require
> `X-Fennec-Client: 1` except for `/api/health`.

### Service commands

After installation, `fennec-agent` invokes the same flattened server module:

| Command | Purpose |
|---|---|
| `python server.py serve` | serve the API and PWA; `--host` and `--port` override defaults |
| `python server.py status --url <base>` | inspect engine health and model ID |
| `python server.py chat <prompt>` | one-shot chat through the supervised service |
| `python server.py agent <task>` | bounded tool loop with explicit write, shell, and web permissions |
| `python server.py acp` | ACP v1 over stdio |
| `python server.py outlets` | show optional outlet status |

## Modules

| Module | Responsibility |
|---|---|
| `server.py` | canonical API/PWA host, loopback guards, static integration |
| `engine.py` | engine lifecycle, health, model discovery, telemetry, embeddings, STT |
| `loop.py` | bounded agent loop, tool-call feedback, step cap |
| `tools.py` | jailed read/edit/grep/run/git tools and optional NPU semantic search |
| `context.py` | payload compaction without truncating the UI transcript |
| `acp.py` | ACP v1 agent, session, and stdio framing |
| `mcp.py`, `mcp_server.py` | external MCP pool and Fennec MCP endpoint |
| `preflight.py` | weights and peak KV-cache checks against available memory |
| `outlets.py` | gated external tasks, transport discovery, jail, host checks, redaction |
| `config.py` | shared read-only JSON configuration and environment overrides |
| `setup_wizard.py` | headless setup and recommended-model download |
| `convert.py` | model-format classification and conversion planning |
| `rust_bridge.py` | optional delegation to the Rust crates |
| `webagent.py`, `web_driver.py`, `snapshot.py` | browser loop and CDP process boundary |
| `liquid.py` | hosted or loopback Liquid decision client |
| `stt.py` | faster-whisper sidecar on `127.0.0.1:8183` |
| `events.py`, `te_bridge.py`, `updater.py`, `version.py` | event support, tokenizer/economy bridge, updates, version contract |

NPU semantic search is off by default. It requires a configured embedding endpoint and explicit
opt-in; Fennec does not probe for an NPU otherwise.

## Rust workspace

The six Rust crates are version 1.0.0, edition 2024, and contain pure logic rather than inference
or model weights. Consumers depend on these implementations rather than copying them.

| Crate | Responsibility |
|---|---|
| `fennec-pyjson` | byte-compatible Python JSON formatting |
| `fennec-acp` | ACP framing, clipping, and piece splitting |
| `fennec-outlets` | secret redaction and blocked-host checks |
| `fennec-preflight` | memory-preflight planning |
| `fennec-client` | loopback clients and suite/manifest discovery |
| `fennec-workspace-tools` | workspace-jailed read/edit/grep/run tools and matching jail paths |

Context compaction has one Rust implementation in
[fennec-economy](https://github.com/Liso7150/fennec-economy), shared by core, desktop, and CLI.
Rust delegation is opt-in with `FENNEC_RUST=1` and a payload-size threshold controlled by
`FENNEC_RUST_MIN_CHARS`.

```bash
cargo build --release --offline
cd src && cargo test --release --offline    # 69 tests across 16 suites
```

## Context economy

The loop compacts only model-bound tool results. It keeps each result's head and tail, replaces
identical results with a reference, and reduces the oldest results to meet the total budget.
It never removes the system prompt, user task, or newest result.

| Variable | Default | Purpose |
|---|---:|---|
| `FENNEC_AGENT_RESULT_CHARS` | 6000 | per-result cap; minimum 600 |
| `FENNEC_AGENT_CONTEXT_BUDGET` | 48000 | model-bound characters per step |

`/api/agent/poll` reports `context_chars_before`, `context_chars_after`, `context_elided`, and
`context_deduped`. A measured local comparison reduced prompt payload by 23.9%, prompt tokens
by 2,599, and median TTFT by 27.7%; this was a two-pair host measurement against a
Colibri-served endpoint, not a general engine latency claim.

## Headless setup

`fennec setup` detects Python, the engine, and models, checks dependencies, and atomically writes
the desktop-compatible configuration. Secrets belong in environment variables, never the JSON
file. An existing configuration is backed up before replacement.

```bash
fennec setup                                  # prompted defaults
fennec setup --yes --download-model           # provision and select the recommended model
fennec setup --dry-run --json                 # print without writing or downloading
fennec setup --yes --install-deps             # install missing core dependencies
```

`--engine-root` must resolve to a checkout containing a built Fennec binary. `backend` accepts
`auto`, `cuda`, `vulkan`, `metal`, `ort`, or `cpu`; `policy` accepts `quality`, `balanced`, or
`experimental-fast`.

## Tests

```bash
python -m unittest discover -s tests -v     # 561 collected: 560 passed, 1 skipped
cd src && cargo test --release --offline    # 69 tests across 16 suites
```

Tests are hermetic: no engine, network, or model is required. Outlet HTTP and subprocesses are
mocked, paid decision calls use fakes, and Rust goldens pin Python-compatible wire formatting.

## Credits

Engine: [Liso7150/fennec](https://github.com/Liso7150/fennec). Fine-tuning lineage:
[AirLLM](https://github.com/lyogavin/airllm). Browser policy uses a Liquid d1 decision model,
hosted or open-weight. The page reader derives from
[browser-use/jev-ultrafast](https://github.com/browser-use/jev-ultrafast); its CDP transport uses
[browser-harness](https://github.com/browser-use/browser-harness) in a separate browser-tools
environment.

Part of the [Fennec suite](https://github.com/Liso7150/fennec-environment-setup).
