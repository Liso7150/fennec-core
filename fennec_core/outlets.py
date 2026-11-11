"""Optional web-agent outlets: Jev (jev-ultrafast) and OpenBot.

Neither is required by Fennec, and neither repo is modified: Jev is driven
through its own ``jev_ultrafast.Agent`` API via ``fennec_core.jev_driver``, and
OpenBot is spoken to over the AG-UI routes it already publishes. Both stay on
this machine; keys stay wherever the upstream put them.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from ipaddress import ip_address
from urllib.parse import quote, urlsplit

from .jev_driver import SENTINEL as JEV_SENTINEL

DEFAULT_JEV_ROOTS = (".fennec/browser-tools/jev-ultrafast", "jev-ultrafast", "Fennec-Suite/jev-ultrafast")
OPENBOT_DEFAULT = "http://127.0.0.1:3001"
OPENBOT_TIMEOUT_S = 900
JEV_TIMEOUT_S = 240
# Fennec's own credentials. A Jev checkout is third-party code on this machine and
# a tool transcript is model-visible, so neither gets them.
SECRET_ENV_KEYS = ("FENNEC_API_KEY", "COLI_API_KEY", "FENNEC_APP_TOKEN")
REDACT_ENV_KEYS = SECRET_ENV_KEYS + ("TYPESAFE_API_KEY",)
JEV_REPORT_BYTES = 200_000
JEV_ERROR_BYTES = 60_000
OPENBOT_EVENT_CAP = 20_000


def _loopback(origin: str) -> bool:
    parts = urlsplit(origin)
    if parts.scheme != "http" or not parts.hostname:
        return False
    if parts.username or parts.password:
        # "http://user:pw@127.0.0.1" reaches 127.0.0.1 but carries a secret to it.
        return False
    if parts.hostname.lower() == "localhost":
        return True
    try:
        return ip_address(parts.hostname).is_loopback
    except ValueError:
        return False


def _blocked_host(url: str) -> str:
    """Refuse the addresses a browser must not be pointed at by a prompt."""
    host = urlsplit(url).hostname or ""
    try:
        address = ip_address(host)
    except ValueError:
        return ""  # a name is resolved by the browser, and may well be the user's own
    if address.is_link_local or address.is_multicast or address.is_reserved \
            or address.is_unspecified:
        return f"web_task will not browse {host}: it is a link-local or reserved address"
    return ""


class _NoRedirects(urllib.request.HTTPRedirectHandler):
    """Never let one loopback answer move the next request off this machine.

    urllib follows 3xx on its own, so an OpenBot that replies "Location: http://host
    /leak" would be re-requested by us and its body read back into a transcript.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _opener():
    return urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirects())


def redact(text: str, environ=None) -> str:
    environ = os.environ if environ is None else environ
    for key in REDACT_ENV_KEYS:
        secret = environ.get(key)
        if secret:
            text = text.replace(secret, "[redacted]")
    return re.sub(r"(?i)(bearer\s+)[A-Za-z0-9._-]{8,}", r"\1[redacted]", text)


class Outlets:
    """Discovery plus two delegating calls. Every failure is a returned error."""

    def __init__(self, config=None, environ=None):
        self.config = config
        self.env = dict(os.environ if environ is None else environ)
        self._cache: dict = {}

    def _cached(self, key, produce, ttl):
        now = time.monotonic()
        stamp, value = self._cache.get(key, (0.0, None))
        if value is not None and now - stamp < ttl:
            return value
        value = produce()
        self._cache[key] = (now, value)
        return value

    # --- discovery --------------------------------------------------------
    def jev_root(self) -> Path | None:
        candidates = []
        configured = ((getattr(self.config, "jev_root", "") if self.config else "")
                      or self.env.get("FENNEC_JEV_ROOT", ""))
        if configured:
            candidates.append(Path(configured).expanduser())
        home = Path.home()
        candidates += [home / relative for relative in DEFAULT_JEV_ROOTS]
        suite = getattr(self.config, "suite_root", None) if self.config else None
        if suite:
            candidates.append(Path(suite) / "jev-ultrafast")
        for candidate in candidates:
            if all((candidate / marker).exists() for marker in ("pyproject.toml", "jev_ultrafast")):
                return candidate
        return None

    def jev_importable(self) -> bool:
        try:
            import importlib.util
            return importlib.util.find_spec("jev_ultrafast") is not None
        except (ImportError, ValueError):
            return False

    def jev_status(self, refresh=False) -> dict:
        if refresh:
            self._cache.pop("jev", None)
        return self._cached("jev", self._jev_probe, 300.0)

    def _jev_probe(self) -> dict:
        root = self.jev_root()
        importable = self.jev_importable()
        return {"present": bool(root or importable),
                "root": str(root) if root else None,
                "importable": importable,
                "launcher": ("uv" if root and shutil.which("uv") else None),
                "policy_key": bool(self.env.get("TYPESAFE_API_KEY")
                                   or (root and (root / ".env").is_file())),
                "paid_api": True,
                "note": "TypeSafe Jev answers CLICK/TYPE_TEXT/SELECT and the page "
                        "element index in one round trip; upstream is never modified."}

    def openbot_base(self) -> str:
        configured = ((getattr(self.config, "openbot_url", "") if self.config else "")
                      or self.env.get("FENNEC_OPENBOT_URL", ""))
        return (configured or OPENBOT_DEFAULT).rstrip("/")

    def openbot_status(self, refresh=False) -> dict:
        if refresh:
            self._cache.pop("openbot", None)
        return self._cached("openbot", self._openbot_probe, 20.0)

    def _openbot_probe(self) -> dict:
        base = self.openbot_base()
        out = {"present": False, "url": base, "loopback": _loopback(base), "agents": []}
        if not _loopback(base):
            out["error"] = ("OpenBot has no machine token for its run API; only a loopback "
                            "single-user server can be driven from here")
            return out
        payload = self._openbot_get(base + "/api/agents")
        if payload.get("error"):
            out["error"] = payload["error"]
            return out
        agents = payload.get("agents")
        if not isinstance(agents, list):
            return out
        out.update({"present": True,
                    "agents": [{"id": item.get("id"), "name": item.get("name")}
                               for item in agents if isinstance(item, dict)][:20]})
        return out

    def available(self, tool_name: str) -> bool:
        if tool_name == "web_task":
            return self.jev_status()["present"]
        if tool_name == "coworker_task":
            return self.openbot_status()["present"]
        return False

    def status(self, refresh=False) -> dict:
        return {"jev": self.jev_status(refresh=refresh),
                "openbot": self.openbot_status(refresh=refresh)}

    # --- Jev --------------------------------------------------------------
    def web_task(self, goal: str, url: str = "", *, max_steps: int = 12,
                 timeout: float = JEV_TIMEOUT_S) -> dict:
        """Hand one URL-scoped end state to the local Jev browser agent."""
        if not isinstance(goal, str) or not goal.strip():
            return {"ok": False, "error": "web_task needs a goal (the end state, not the hops)"}
        root = self.jev_root()
        if not (root or self.jev_importable()):
            # Discovery said present up to five minutes ago and the checkout is gone
            # (or never was); let the next look be a real one.
            self._cache.pop("jev", None)
            return {"ok": False, "error": ("no jev-ultrafast checkout or module on this machine; "
                                           "clone https://github.com/browser-use/jev-ultrafast, "
                                           "or leave the outlet unconfigured")}
        url = (url or "").strip() or "https://www.google.com/"
        if not url.lower().startswith(("http://", "https://")):
            return {"ok": False, "error": "web_task needs an http(s) start url"}
        refused = _blocked_host(url)
        if refused:
            return {"ok": False, "error": refused}
        command, cwd, env = self._jev_command(root, url, goal.strip(), max_steps, timeout)
        if command is None:
            return {"ok": False, "error": "no interpreter can run the Jev driver"}
        stdout, stderr, outcome = self._run_jev(command, cwd, env, max(5, float(timeout)) + 15)
        if outcome == "timeout":
            self._cache.pop("jev", None)
            return {"ok": False, "error": f"jev outlet timed out after {timeout}s"}
        if outcome == "failed-to-start":
            return {"ok": False, "error": f"jev outlet could not start: {stderr}"}
        report = _report(stdout)
        if report is None:
            return {"ok": False,
                    "error": redact((stderr or stdout or "").strip()[-2000:], self.env),
                    "returncode": outcome}
        report.setdefault("ok", outcome == 0)
        if outcome != 0 and report.get("ok"):
            report["ok"] = False
            report["error"] = f"the Jev driver exited {outcome} after reporting success"
        report["outlet"] = "jev-ultrafast"
        return report

    def _run_jev(self, command, cwd, env, timeout):
        """Run the driver with files behind the pipes instead of pipes themselves.

        capture_output=True makes subprocess drain the pipes in its own threads, and
        on Windows a browser grandchild that inherits them keeps them open: the
        TimeoutExpired branch then blocks in communicate() for as long as that
        grandchild lives, which is longer than the timeout we just enforced.
        """
        options = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
        with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
            try:
                done = subprocess.run(command, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                                      stdout=out, stderr=err, timeout=timeout, check=False,
                                      **options)
            except subprocess.TimeoutExpired:
                return "", "", "timeout"
            except OSError as exc:
                return "", redact(str(exc), env), "failed-to-start"
            out.seek(0)
            err.seek(0)
            return (out.read(JEV_REPORT_BYTES).decode("utf-8", "replace"),
                    err.read(JEV_ERROR_BYTES).decode("utf-8", "replace"), done.returncode)

    def _jev_command(self, root, url, goal, max_steps, timeout):
        driver = str(Path(__file__).with_name("jev_driver.py"))
        env = {key: value for key, value in self.env.items() if key not in SECRET_ENV_KEYS}
        if root:
            env.setdefault("PYTHONPATH", str(root))
        if root and shutil.which("uv"):
            return (["uv", "run", "--quiet", "--project", str(root), sys.executable, driver,
                     "--url", url, "--goal", goal, "--max-steps", str(int(max_steps or 12)),
                     "--timeout", str(int(timeout))]), str(root), env
        return [sys.executable, driver, "--url", url, "--goal", goal,
                "--max-steps", str(int(max_steps or 12)),
                "--timeout", str(int(timeout))], (str(root) if root else None), env

    # --- OpenBot ----------------------------------------------------------
    def coworker_task(self, goal: str, agent: str = "", *, thread_id: str = "",
                      timeout: float = OPENBOT_TIMEOUT_S) -> dict:
        """Ask one OpenBot coworker for a run and read its AG-UI stream back."""
        if not isinstance(goal, str) or not goal.strip():
            return {"ok": False, "error": "coworker_task needs a goal"}
        state = self.openbot_status()
        if not state["present"]:
            return {"ok": False, "error": state.get("error") or
                    "no OpenBot workspace is answering on " + state["url"]}
        agent_id = self._openbot_agent_id(state["agents"], agent)
        if isinstance(agent_id, dict):
            return agent_id
        body = {"threadId": thread_id or str(uuid.uuid4()), "runId": str(uuid.uuid4()),
                "state": {}, "tools": [], "context": [], "forwardedProps": {},
                "messages": [{"id": str(uuid.uuid4()), "role": "user", "content": goal.strip()}]}
        return self._openbot_run(state["url"], agent_id, body, timeout)

    def _openbot_agent_id(self, agents, wanted: str):
        if isinstance(wanted, str) and wanted.strip():
            wanted = wanted.strip()
            agent_id = next((item.get("id") for item in agents
                             if wanted in {str(item.get("id")), str(item.get("name"))}), None)
            if agent_id is None:
                return {"ok": False, "error": f"unknown OpenBot coworker {wanted!r}",
                        "agents": agents}
        elif len(agents) == 1:
            agent_id = agents[0].get("id")
        else:
            return {"ok": False,
                    "error": ("OpenBot answers but lists no coworker agent" if not agents
                              else "OpenBot has more than one coworker; name one with agent="),
                    "agents": agents}
        if not isinstance(agent_id, (str, int)) or not str(agent_id).strip():
            return {"ok": False, "error": "the OpenBot coworker has no usable id", "agents": agents}
        return quote(str(agent_id), safe="")

    def _openbot_run(self, base: str, agent_id: str, body: dict, timeout: float) -> dict:
        request = urllib.request.Request(
            f"{base}/api/copilotkit/agent/{agent_id}/run",
            data=json.dumps(body).encode(),
            headers={"content-type": "application/json", "accept": "text/event-stream"})
        text, events, error, overrun = [], [], None, ""
        limit = max(5.0, float(timeout))
        deadline = time.monotonic() + limit
        finished = False
        try:
            with _opener().open(request, timeout=limit) as response:
                for event in _sse_events(response):
                    events.append(event.get("type"))
                    if event.get("type") == "TEXT_MESSAGE_CONTENT":
                        text.append(redact(str(event.get("delta") or ""), self.env))
                    elif event.get("type") == "RUN_ERROR":
                        error = redact(str(event.get("message") or "OpenBot run failed"), self.env)
                        break
                    elif event.get("type") == "RUN_FINISHED":
                        finished = True
                        break
                    # The socket timeout only bounds the gap between two events; a
                    # coworker that streams forever needs a wall clock as well.
                    if len(events) >= OPENBOT_EVENT_CAP:
                        overrun = f"the OpenBot stream carried more than {OPENBOT_EVENT_CAP} events"
                        break
                    if time.monotonic() > deadline:
                        overrun = f"the OpenBot stream outlived {int(limit)}s"
                        break
        except urllib.error.HTTPError as exc:
            if exc.code in (301, 302, 303, 307, 308):
                return {"ok": False, "outlet": "openbot",
                        "error": f"OpenBot answered HTTP {exc.code} with a redirect, which this "
                                 "outlet refuses to follow off the configured loopback server"}
            return {"ok": False, "outlet": "openbot", "error": f"OpenBot run refused: HTTP {exc.code}"}
        except (urllib.error.URLError, OSError):
            # Mid-run the workspace went away; do not keep advertising it for the
            # rest of the cache lifetime.
            self._cache.pop("openbot", None)
            return {"ok": False, "outlet": "openbot",
                    "error": "OpenBot run request failed: the workspace stopped answering"}
        answer = "".join(text).strip()
        if error:
            return {"ok": False, "outlet": "openbot", "error": error, "events": events[-8:]}
        if overrun:
            return {"ok": False, "outlet": "openbot", "error": overrun, "events": events[-8:],
                    "answer": answer[:8000]}
        if not finished and not answer:
            return {"ok": False, "outlet": "openbot", "error": "stream ended without RUN_FINISHED",
                    "events": events[-8:]}
        return {"ok": True, "outlet": "openbot", "thread": body["threadId"],
                "run": body["runId"], "answer": answer[:8000], "steps": len(events),
                "complete": finished}

    def _openbot_get(self, url: str):
        try:
            with _opener().open(urllib.request.Request(url, headers={"accept": "application/json"}),
                                timeout=5) as response:
                value = _json_value(response.read(2_000_001).decode("utf-8", "replace"))
        except urllib.error.HTTPError as exc:
            if exc.code in (301, 302, 303, 307, 308):
                return {"error": f"HTTP {exc.code}: OpenBot redirected, and this outlet never "
                                 "follows a redirect off the loopback server it was given"}
            hint = ("needs a Better-Auth session; OpenBot issues no machine token for "
                    "its run API" if exc.code in (401, 403) else f"HTTP {exc.code}")
            return {"error": hint}
        except (urllib.error.URLError, OSError):
            return {"error": "no OpenBot workspace answering"}
        if isinstance(value, list):
            return {"agents": value}
        return value if isinstance(value, dict) else {}

def _json_value(raw):
    try:
        return json.loads(raw or "")
    except (ValueError, TypeError):
        return None


def _report(raw):
    """The driver's JSON, taken from after its sentinel instead of all of stdout.

    Jev owns its own logging, and a banner or a progress line on stdout would
    otherwise turn a finished, paid browser run into "the outlet printed no JSON".
    Anything on either side of the object is noise, so read one value and stop.
    """
    marker = (raw or "").rfind(JEV_SENTINEL)
    if marker < 0:
        return None
    try:
        value, _ = json.JSONDecoder().raw_decode(raw[marker + len(JEV_SENTINEL):].lstrip())
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def _sse_events(lines):
    """Yield decoded JSON objects from an SSE byte/line stream."""
    for raw in lines:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", "replace")
        line = raw.strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            event = json.loads(payload)
        except ValueError:
            continue
        if isinstance(event, dict) and event.get("type"):
            yield event
