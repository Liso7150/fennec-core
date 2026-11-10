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

## Context economy (1.4.1)

The agent loop compacts only the payload it sends to the model; the transcript the
UI shows is untouched. A tool result keeps its head and tail, so a cut JSON blob
no longer loses its end; a repeated result becomes
`[identical to the result of <call>; N chars not repeated]`; and a total budget
shrinks the oldest tool results first, never the system prompt, the user task or
the newest result.

| Variable | Default | Meaning |
|---|---|---|
| `FENNEC_AGENT_RESULT_CHARS` | 6000 | per-result cap (floor 600) |
| `FENNEC_AGENT_CONTEXT_BUDGET` | 48000 | characters sent per step (never below the result cap) |

Savings are reported per session in `/api/agent/poll` usage as
`context_chars_before`, `context_chars_after`, `context_elided` and
`context_deduped`. Measured on a representative transcript with the defaults:
28,117 -> 18,755 characters per step (33.3 % fewer).
