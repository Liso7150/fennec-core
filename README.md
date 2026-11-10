# Colibri Core

Shared core for the Colibri apps: an engine client/supervisor, an agent loop,
tool plugins with a path jail, and an Intel NPU embedding sidecar. Extracted from
the duplicated logic that previously lived in the desktop (JS), coding-agent
(Python) and JetBrains (Java) apps so there is one reference implementation.

## Features

- Attach-first engine supervisor: spawns on demand, stops the child it owns with all descendants, never touches an attached server.
- Supervised agent loop with a step cap, tool-result feedback and explicit write/shell approvals.
- Workspace-jailed tools (read, edit, grep, run, git) plus NPU semantic search.
- FastAPI service that hosts the PWA with loopback Host/Origin checks, an optional app token and TLS.
- One shared config contract (`~/.colibri/config.json`, `COLIBRI_*` overrides) for every app in the suite.

## Modules

- `colibri_core/engine.py` — spawn/stop the Colibri engine, health checks,
  model-id discovery, telemetry (`/metrics`, `/experts`, `/profile`).
- `colibri_core/loop.py` — bounded agent loop (tool-call accumulation, result
  feedback, step cap).
- `colibri_core/tools.py` — file/edit/grep/run/git tools with a workspace jail,
  plus NPU semantic search.
- `colibri_core/server.py` — FastAPI agent server.

## Credits

Built on the [Colibri engine](https://github.com/JustVugg/colibri) and
[AirLLM](https://github.com/lyogavin/airllm). Part of the
[Colibri suite](https://github.com/Liso7150/colibri-environment-setup).