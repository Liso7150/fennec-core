"""Local speech-to-text sidecar: faster-whisper on the CPU, loopback only.

Started on demand by colibri_core.engine (phone PWA and desktop both use it
through POST /api/stt). The model is downloaded once into the shared HF cache
(HF_HOME under cache_root) by faster-whisper itself; audio never leaves the
machine.

    python -m colibri_core.stt --port 8183 [--model base.en]

Endpoints: GET /health, POST /transcribe (raw audio body: webm/ogg/wav/mp4).
"""
from __future__ import annotations

import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MAX_BYTES = 16 << 20
DEFAULT_MODEL = os.environ.get("COLI_STT_MODEL", "base.en")
ADMISSIBLE = {"127.0.0.1", "::1", "localhost"}

_model = None
_model_name = DEFAULT_MODEL


def _load(model_name=None):
    global _model, _model_name
    if _model is None:
        from faster_whisper import WhisperModel
        _model_name = model_name or _model_name
        threads = max(2, min(8, os.cpu_count() or 2))
        _model = WhisperModel(_model_name, device="cpu", compute_type="int8",
                              cpu_threads=threads)
    return _model


def transcribe_bytes(data: bytes, language: str | None = None) -> dict:
    import io
    model = _load()
    audio = io.BytesIO(data)
    options = {"beam_size": 1, "vad_filter": True, "condition_on_previous_text": False}
    if language:
        options["language"] = language
    segments, info = model.transcribe(audio, **options)
    text = "".join(segment.text for segment in segments).strip()
    return {"text": text, "language": getattr(info, "language", None),
            "duration_s": round(float(getattr(info, "duration", 0) or 0), 2)}


class Handler(BaseHTTPRequestHandler):
    server_version = "ColibriSTT/1.0"

    def _reply(self, code, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _loopback(self):
        host = self.client_address[0] if self.client_address else ""
        return host in ADMISSIBLE or host.startswith("127.")

    def do_GET(self):
        if not self._loopback():
            return self._reply(403, {"error": "loopback only"})
        if self.path == "/health":
            return self._reply(200, {"status": "ok", "model": _model_name, "device": "cpu",
                                     "loaded": _model is not None})
        return self._reply(404, {"error": "not found"})

    def do_POST(self):
        if not self._loopback():
            return self._reply(403, {"error": "loopback only"})
        if self.path != "/transcribe":
            return self._reply(404, {"error": "not found"})
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return self._reply(400, {"error": "invalid Content-Length"})
        if length <= 0 or length > MAX_BYTES:
            return self._reply(413, {"error": "audio must be 1 byte to 16 MiB"})
        data = self.rfile.read(length)
        try:
            result = transcribe_bytes(data)
        except Exception as exc:  # decoding/model errors are client-visible text
            return self._reply(422, {"error": f"could not transcribe audio: {exc}"})
        return self._reply(200, result)

    def log_message(self, *_args):
        pass


def serve(port=8183, model_name=DEFAULT_MODEL, host="127.0.0.1"):
    global _model_name
    _model_name = model_name
    _load(model_name)  # Fail fast: a broken download must not look like a dead server.
    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    server.serve_forever()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Local Colibri speech-to-text sidecar")
    parser.add_argument("--port", type=int, default=int(os.environ.get("COLI_STT_PORT", "8183")))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args(argv)
    serve(args.port, args.model, args.host)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
