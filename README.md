# Fennec Core

Shared core for the Fennec apps: an engine client/supervisor, an agent loop,
tool plugins with a path jail, and an Intel NPU embedding sidecar. Extracted from
the duplicated logic that previously lived in the desktop (JS), coding-agent
(Python) and JetBrains (Java) apps so there is one reference implementation.

## Features

- Attach-first engine supervisor: spawns on demand, stops the child it owns with all descendants, never touches an attached server.
- Supervised agent loop with a step cap, tool-result feedback and explicit write/shell approvals.
- Reasoning pass-through: `reasoning_effort`/`enable_thinking` requests and `reasoning_content` deltas/steps.
- Vision content parts: one `data:image/...;base64` image per request, validated before it reaches the engine.
- Local speech sidecar: faster-whisper on CPU (`whisper base.en`, int8) behind `POST /api/stt` on `127.0.0.1:8183`, started on demand. Install with `pip install "fennec-core[stt]"` (or `faster-whisper==1.2.1 av>=14,<18`; PyAV 18+ broke faster-whisper 1.2.1's `metadata_errors`).
- Workspace-jailed tools (read, edit, grep, run, git) plus NPU semantic search.
- FastAPI service that hosts the PWA with loopback Host/Origin checks, an optional app token and TLS.
- One shared config contract (`~/.fennec/config.json`, `FENNEC_*` overrides) for every app in the suite.

## Modules

- `fennec_core/engine.py` — spawn/stop the Colibri engine, health checks,
  model-id discovery, telemetry (`/metrics`, `/experts`, `/profile`), STT sidecar lifecycle.
- `fennec_core/loop.py` — bounded agent loop (tool-call accumulation, result
  feedback, step cap).
- `fennec_core/tools.py` — file/edit/grep/run/git tools with a workspace jail,
  plus NPU semantic search.
- `fennec_core/server.py` — FastAPI agent server, chat with reasoning/vision, `/api/stt`.
- `fennec_core/stt.py` — local faster-whisper transcription sidecar.

## Credits

Engine base idea: [JustVugg/colibri](https://github.com/JustVugg/colibri) —
device-tiered MoE inference. Fine-tuning:
[AirLLM](https://github.com/lyogavin/airllm). Part of the
[Fennec suite](https://github.com/Liso7150/fennec-environment-setup).