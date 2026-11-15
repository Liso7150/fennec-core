# Changelog

All notable changes to fennec-core, the shared runtime for the Fennec suite.

The sections below carry the published tag each change actually shipped in
(`Liso7150/fennec-core`, `v0.1.0` … `v0.9.0`). An earlier revision of this file
numbered the same history on a 1.x/2.x ladder that was never published; those
tags still exist locally (`v1.2.0` … `v2.0.0`) and each one points at the same
commit as a published `v0.x` tag. `v0.2.0` was never released here — the ladder
goes `0.1.0` → `0.3.0`. Dates are the tag publish dates.

## [Unreleased] — 1.0.0 in preparation

Nothing at 1.0.0 is published. The newest release here is `v0.9.0`
(2026-11-11T19:29Z). Measured on the working tree: 251 Python tests collected
(`python -m pytest --collect-only -q`, 2026-11-15) and 53 `#[test]` functions
across the 6 Rust crates in `src/`.

### Added
- Typed event envelope system with domain models in Rust crates
- ACP agent improvements and expanded test coverage
- `fennec-client` parses the outlet status instead of guessing at it
- CI workspace tests expanded to macOS and Ubuntu
- `fennec-napi` workspace member (Node-API bridge), committed so the CI
  workspace loads

### Changed
- Every engine knob spelled `FENNEC_*` first, with `COLI_*` kept as alias
- Preflight goldens carry their own fixture
- Rust workspace moved from `rust/` to `src/`
- Desktop shell moved out to the fennec-desktop repo; the CLI moved out to
  fennec-cli; workflows for the moved crates retired
- The Rust desktop shell (SSE), the WebView2 IPC shell and the token-stream CI
  matrix landed here first and moved with the desktop shell
- The pipe section quotes measured numbers, not the pre-gate ones

### Fixed
- Engine named correctly in the 503 the desktop app shows
- Token-stream matrix workflow (block YAML, winner script)
- `wry 0.57` builder API (`new()` + `build(&window)`)
- API route slashes normalized to handler names, and the CI smoke URL's trailing
  slash

## [0.9.0] - 2026-11-11

### Added
- Rust port of the suite glue: outlets gate, preflight math, ACP framing crates

### Changed
- The pipe only pays at 8 MB (measured)

## [0.8.0] - 2026-11-11

### Changed
- One ACP v1 agent for every front end

## [0.7.1] - 2026-11-11

### Added
- `fennec setup` terminal wizard (`setup_wizard.py` plus its tests)

## [0.7.0] - 2026-11-11

Version bump only — no change shipped between `0.6.0` and `0.7.0`.

## [0.6.0] - 2026-11-11

### Added
- Model preflight check

## [0.5.0] - 2026-11-11

### Added
- Optional TokenEconomizer bridge: learned facts, loop guard, in-process fold

## [0.4.1] - 2026-11-11

### Changed
- Resolve python from PATH instead of an engine venv

## [0.4.0] - 2026-11-10

### Changed
- Default 4 GB host expert pool on backend auto
- Request the Vulkan tier when backend auto finds a loader

## [0.3.1] - 2026-11-10

### Added
- Agent payload compaction: elide, dedupe and budget tool results

### Changed
- Shared config roots default to the user home instead of a fixed drive

## [0.3.0] - 2026-11-10

### Changed
- Rebrand the shared runtime to Fennec

## [0.1.0] - 2026-11-10

The first published release. It carries the work described in the commits below,
which were tagged ahead of the release rather than separate releases.

### Added
- Reasoning content parts
- Vision content parts
- On-demand STT (speech-to-text) sidecar
- Secure FastAPI service, agent loop, packaged CLI, 24 tests
- Shared runtime for the suite, committed as colibri-core on 2026-11-09
- Credits: JustVugg/colibri + lyogavin/airllm

### Changed
- Features documented in the README

### Fixed
- Stop an engine the service spawned when the core shuts down
