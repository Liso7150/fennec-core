"""Optional agent outlets: Fennec's own browser loop, OpenBot and Antigravity.

None is required by Fennec. ``web_task`` is answered by Fennec's own loop
(:mod:`webagent`), which asks Liquid d1.2 for the next operation and the local engine
for its wording; the loop is driven from here, in a child process, through the browser
transport it already uses. OpenBot is spoken to over the AG-UI routes it already
publishes. Everyone stays on this machine; keys stay wherever the owner put them.
"""
from __future__ import annotations

import asyncio
import importlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from ipaddress import ip_address
from urllib.parse import quote, urlsplit

import liquid
from web_driver import SENTINEL as WEB_SENTINEL

OPENBOT_DEFAULT = "http://127.0.0.1:3001"
OPENBOT_TIMEOUT_S = 900
WEB_TIMEOUT_S = 240
WEB_PYTHON_ENV = "FENNEC_WEB_PYTHON"
# The transport the loop drives Chrome with. It is not a dependency of core, so the loop
# runs in whichever interpreter has it installed and imports core back through PYTHONPATH.
WEB_TRANSPORT = "browser_harness.helpers"
# Where Fennec keeps its own copy of that transport: a venv this suite owns, beside the
# checkpoints and caches, so the browser lane needs nothing but Fennec to run.
BROWSER_ENV = (".fennec", "browser-tools", "fennec")
# How long an interpreter is given to answer "do you have the transport". It is one import
# of a small pure-python package; anything slower than this is a machine problem, and the
# answer ("no") is the same one a timeout would give.
BROWSE_PROBE_S = 10.0
# A lane gets its own timeout, then this much longer to die. The driver enforces the first
# and prints its report; the second is for one that a browser grandchild has wedged, where
# killing it is the only answer available.
LANE_GRACE_S = 15.0
# Fennec's own credentials. A tool transcript is model-visible and a child process
# inherits the environment, so these stay out of both.
SECRET_ENV_KEYS = ("FENNEC_API_KEY", "COLI_API_KEY", "FENNEC_APP_TOKEN", "FENNEC_MCP_TOKEN",
                   "LIQUID_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY",
                   "GOOGLE_APPLICATION_CREDENTIALS")
BROWSER_PROXY_ENV_KEYS = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
                          "http_proxy", "https_proxy", "all_proxy", "NODE_USE_ENV_PROXY")
WEB_REPORT_BYTES = 200_000
ANTIGRAVITY_TIMEOUT_S = 900
WEB_ERROR_BYTES = 60_000
# A live browser lane is polled, not waited on, so that the notes it appends can reach the
# chat while a decision is still running. This is how long the parent sleeps between looks.
DRIVER_POLL_S = 0.2
# How long a driver that ran out of its grace period gets to die before its report is read
# from the files anyway.
DRIVER_KILL_GRACE_S = 5.0
# Notes are short lines a handful at a time; this only bounds one pass of the tail.
NOTES_BYTES = 65_536
OPENBOT_EVENT_CAP = 20_000


def _split(url: str):
    """`urlsplit`, or None when the netloc will not parse at all.

    urllib raises for a bracketed literal that is not an address, and a browser
    request built from model text must never turn into an exception - every
    failure in this module is a returned error.
    """
    try:
        return urlsplit(url)
    except ValueError:
        return None


def _same_file(left: str, right: str) -> bool:
    """Whether two spellings name the same program.

    The interpreter this process runs in is asked about in-process and never spawned, so
    recognising it through a venv symlink or a different case is what keeps a discovery
    probe from paying a process start to ask a question it can already answer.
    """
    try:
        return bool(left) and bool(right) and Path(left).resolve() == Path(right).resolve()
    except OSError:
        return False


def _loopback(origin: str) -> bool:
    parts = _split(origin)
    if parts is None or parts.scheme != "http" or not parts.hostname:
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
    parts = _split(url)
    if parts is None:
        return ("web_task will not browse that url: its host is not a parseable "
                "address")
    host = parts.hostname or ""
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


def _bypass_list(value: str) -> str:
    """A ``NO_PROXY`` rewritten the way httpx reads it.

    httpx parses every entry as a URL authority, so the bracketed IPv6 spelling that curl
    accepts raises ``InvalidURL`` inside ``httpx.Client()`` - and the browser transport builds
    that client while it is being imported, so the lane dies before it reaches the page. A
    parent process can hand us the bracketed form even when the persistent value on the
    machine is the bare address, so the entry is unbracketed rather than dropped: the bypass
    survives, spelled once.
    """
    entries = []
    for entry in value.split(","):
        entry = entry.strip()
        if entry.startswith("[") and entry.endswith("]"):
            entry = entry[1:-1]
        if entry and entry not in entries:
            entries.append(entry)
    return ",".join(entries)


def _opener():
    return urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirects())


def redact(text: str, environ=None) -> str:
    """Fennec's own credentials, then any bearer token.

    The Rust port does the identical work - `fennec-rs redact` and the golden in
    `src/goldens/redact.jsonl` pin it - and is faster at the batch, but every
    size measured costs more through the pipe than here, so the live path stays
    in-process. See `rust_bridge.py` for the numbers.
    """
    environ = os.environ if environ is None else environ
    for key in SECRET_ENV_KEYS:
        secret = environ.get(key)
        if secret:
            text = text.replace(secret, "[redacted]")
    return re.sub(r"(?i)(bearer\s+)[A-Za-z0-9._-]{8,}", r"\1[redacted]", text)


def _antigravity_sdk():
    """Load the optional SDK and its policy hooks without making it a core dependency."""
    try:
        module = importlib.import_module("google.antigravity")
        hooks = importlib.import_module("google.antigravity.hooks")
    except (ImportError, ModuleNotFoundError):
        return None
    return module, hooks.policy


class Outlets:
    """Discovery plus optional delegating calls. Every failure is a returned error."""

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
    def browser_env_python(self) -> str:
        """The interpreter inside Fennec's own browser venv, or "" when it has none.

        The loop's transport is a dependency of the lane, not of core, so it belongs in a venv
        this suite owns beside the checkpoints and caches - the alternative was reaching into
        somebody else's checkout for it, which made deleting that checkout take this lane with
        it.
        """
        bins = ("Scripts", "python.exe") if os.name == "nt" else ("bin", "python")
        venv = Path.home().joinpath(*BROWSER_ENV, ".venv", *bins)
        return str(venv) if venv.is_file() else ""

    def web_python(self) -> str:
        """The interpreter told to run the loop, or "" for "discover one"."""
        return (((getattr(self.config, "web_python", "") if self.config else "")
                 or self.env.get(WEB_PYTHON_ENV, ""))).strip()

    def web_candidates(self) -> list:
        """Interpreters that could run the loop, best guess first.

        A pin comes first, then this interpreter - a machine whose core already has the
        transport needs no second venv - and last Fennec's own browser venv, which is the one
        the browser-outlets feature fills. :meth:`web_transport` names in
        :meth:`fennec_status` which of them answered.
        """
        found = [self.web_python()] if self.web_python() else []
        found.append(sys.executable)
        own = self.browser_env_python()
        if own:
            found.append(own)
        return list(dict.fromkeys(found))

    def web_transport(self, python: str) -> str:
        """Where the interpreter running the loop got its browser transport from."""
        if not python:
            return "none"
        if _same_file(python, self.web_python()):
            return "pinned"
        if _same_file(python, self.browser_env_python()):
            return "the Fennec browser env"
        if _same_file(python, sys.executable):
            return "this interpreter"
        return "external"

    def _can_browse(self, python: str) -> bool:
        """Whether one interpreter can import the transport the loop drives Chrome with."""
        if not python:
            return False
        if _same_file(python, sys.executable):
            try:
                return importlib.util.find_spec(WEB_TRANSPORT) is not None
            except (ImportError, ValueError):
                return False
        return self._cached(f"browse:{python}", lambda: self._probe_browse(python), 300.0)

    def _probe_browse(self, python: str) -> bool:
        """Ask another interpreter, once, whether it has the transport.

        A missing, half-installed or unbearably slow interpreter all answer "no", which is
        the same thing an error would mean; the caller cannot act on the difference.
        """
        _, _, outcome = self._run_driver([python, "-c", f"import {WEB_TRANSPORT}"],
                                         None, dict(self.env), BROWSE_PROBE_S)
        return outcome == 0

    def web_interpreter(self) -> str:
        """The first candidate that can really import the transport, or "" for none."""
        return self._cached("web-python", lambda: next(
            (python for python in self.web_candidates() if self._can_browse(python)), ""), 300.0)

    def fennec_status(self, refresh=False) -> dict:
        if refresh:
            self._cache.pop("fennec", None)
        return self._cached("fennec", self._fennec_probe, 300.0)

    def _fennec_probe(self) -> dict:
        python = self.web_interpreter()
        return {"present": bool(python) and self._decision_ready(),
                "interpreter": python or None,
                "transport": bool(python),
                "transport_from": self.web_transport(python),
                "decision_key": bool(self._liquid_key()),
                "model": liquid.model_name(self.env),
                "note": "Fennec's own loop: d1.2 names the operation and the node, the local "
                        "engine words the values, and a mutation is never replayed."}

    def _decision_ready(self) -> bool:
        """Whether anything can answer a decision: a hosted key, or a server on this box."""
        if liquid.is_local(liquid.endpoint_for(self.env)):
            return True
        return bool(self._liquid_key())

    def _liquid_key(self) -> str:
        """The d1.2 key, or "" for "there is none". ``liquid.api_key`` raises instead."""
        try:
            return liquid.api_key(self.env)
        except liquid.LiquidError:
            return ""

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

    # --- Antigravity -------------------------------------------------------
    def antigravity_workspace(self) -> str:
        configured = ((getattr(self.config, "workspace", "") if self.config else "")
                      or self.env.get("FENNEC_WORKSPACE", ""))
        if not configured:
            return ""
        try:
            root = Path(configured).expanduser().resolve()
        except (OSError, RuntimeError, ValueError):
            return ""
        return str(root) if root.is_dir() else ""

    def _antigravity_credentials(self) -> str:
        if self.env.get("GEMINI_API_KEY") or self.env.get("GOOGLE_API_KEY"):
            return "gemini"
        vertex = self.env.get("GOOGLE_GENAI_USE_VERTEXAI", "").casefold() in {"1", "true", "yes"}
        if vertex and self.env.get("GOOGLE_CLOUD_PROJECT") and self.env.get("GOOGLE_CLOUD_LOCATION"):
            return "vertex"
        return ""

    def antigravity_status(self, refresh=False) -> dict:
        if refresh:
            self._cache.pop("antigravity", None)
        return self._cached("antigravity", self._antigravity_probe, 300.0)

    def _antigravity_probe(self) -> dict:
        sdk = _antigravity_sdk()
        workspace = self.antigravity_workspace()
        credential = self._antigravity_credentials()
        out = {"present": bool(sdk and credential and workspace),
               "installed": sdk is not None,
               "configured": bool(credential),
               "workspace": bool(workspace),
               "model": "Google Antigravity",
               "tools": ["list_dir", "read_file", "grep"]}
        if sdk is None:
            out["error"] = "install the optional antigravity extra"
        elif not credential:
            out["error"] = "set GEMINI_API_KEY or configure Vertex AI"
        elif not workspace:
            out["error"] = "set FENNEC_WORKSPACE to an existing directory"
        else:
            out["note"] = "read-only Fennec workspace tools; Antigravity built-ins are disabled"
        return out

    def antigravity_task(self, goal: str, *, timeout: float = ANTIGRAVITY_TIMEOUT_S) -> dict:
        """Run one Antigravity task with only Fennec's jailed read tools."""
        if not isinstance(goal, str) or not goal.strip():
            return {"ok": False, "outlet": "antigravity", "error": "antigravity_task needs a goal"}
        state = self.antigravity_status()
        if not state["present"]:
            return {"ok": False, "outlet": "antigravity",
                    "error": state.get("error") or "Antigravity is not available"}
        sdk_info = _antigravity_sdk()
        if sdk_info is None:
            self._cache.pop("antigravity", None)
            return {"ok": False, "outlet": "antigravity", "error": "Antigravity SDK is not installed"}
        module, policy = sdk_info
        workspace = self.antigravity_workspace()
        from tools import ToolContext, grep, list_dir, read_file

        ctx = ToolContext(workspace=workspace)

        def list_dir_tool(path: str = ".") -> dict:
            try:
                return list_dir(ctx, path)
            except (OSError, PermissionError, ValueError) as exc:
                return {"ok": False, "error": str(exc)}

        def read_file_tool(path: str, numbered: bool = True) -> dict:
            try:
                return read_file(ctx, path, numbered=numbered)
            except (OSError, PermissionError, ValueError) as exc:
                return {"ok": False, "error": str(exc)}

        def grep_tool(pattern: str, path: str = ".") -> dict:
            try:
                return grep(ctx, pattern, path)
            except (OSError, PermissionError, ValueError) as exc:
                return {"ok": False, "error": str(exc)}

        tools = [list_dir_tool, read_file_tool, grep_tool]
        for name, tool in zip(("list_dir", "read_file", "grep"), tools):
            tool.__name__ = name
        try:
            config = module.LocalAgentConfig(
                tools=tools,
                policies=[policy.deny_all(), *(policy.allow(tool.__name__) for tool in tools)],
            )

            async def run():
                async with module.Agent(config) as agent:
                    response = await agent.chat(goal.strip())
                    return await response.text()

            answer = asyncio.run(asyncio.wait_for(run(), timeout=max(5.0, float(timeout))))
        except asyncio.TimeoutError:
            return {"ok": False, "outlet": "antigravity",
                    "error": f"Antigravity did not finish inside {int(timeout)}s"}
        except Exception as exc:
            return {"ok": False, "outlet": "antigravity",
                    "error": redact(f"Antigravity run failed: {exc}", self.env)}
        return {"ok": True, "outlet": "antigravity", "engine": "google-antigravity",
                "answer": redact(str(answer), self.env)[:8000], "tools": [tool.__name__ for tool in tools]}

    def available(self, tool_name: str) -> bool:
        """Whether this machine can answer one tool.

        This is the question the agent loop asks before advertising ``web_task``. Promising a
        browse that refuses the moment it is asked is worse than having no browser tool, so
        the lane is advertised only when both halves of it stand: an interpreter that can
        import the transport, and something to decide with - a hosted d1.2 key or a local
        System One server. ``/api/outlets/run``, where the
        user clicked the row, asks the same question and gets the same answer - there is one
        browser lane, so it has no second spelling.
        """
        if tool_name == "web_task":
            return self.fennec_status()["present"]
        if tool_name == "coworker_task":
            return self.openbot_status()["present"]
        if tool_name == "antigravity_task":
            return self.antigravity_status()["present"]
        return False

    def status(self, refresh=False) -> dict:
        return {"fennec": self.fennec_status(refresh=refresh),
                "openbot": self.openbot_status(refresh=refresh),
                "antigravity": self.antigravity_status(refresh=refresh)}

    # --- the web outlet ---------------------------------------------------
    def web_task(self, goal: str, url: str = "", *, max_steps: int = 12,
                 timeout: float = WEB_TIMEOUT_S, progress=None) -> dict:
        """Hand one URL-scoped end state to Fennec's own browser loop.

        A task this loop cannot start comes back saying why. There is no other agent to spend
        it on: the alternative was handing the page to a third-party agent the user never
        chose, after a first one had already clicked on it.

        ``progress`` is called with ``{"text", "step", "of"}`` as each step begins, which is
        the only way a chat can show movement during a decision that takes half a minute.
        """
        if not isinstance(goal, str) or not goal.strip():
            return {"ok": False, "error": "web_task needs a goal (the end state, not the hops)"}
        url = (url or "").strip() or "https://www.google.com/"
        if not url.lower().startswith(("http://", "https://")):
            return {"ok": False, "error": "web_task needs an http(s) start url"}
        blocked = _blocked_host(url)
        if blocked:
            return {"ok": False, "error": blocked}
        report, refused = self._web_fennec(url, goal.strip(), max_steps, timeout, progress)
        if report is not None:
            return report
        return {"ok": False, "engine": "liquid-d1.2", "outlet": "fennec-webagent",
                "error": refused}

    def _web_fennec(self, url, goal, max_steps, timeout, progress=None):
        """Run Fennec's own loop: ``(report, "")``, or ``(None, why it never started)``.

        None means no browser was reached - an interpreter that cannot import the transport,
        or a caller with nothing to decide with - so nothing has been spent on the page and
        the caller may still say "no" cleanly. A run that timed out has spent something, so
        its own report comes back as it stands.
        """
        python = self.web_interpreter()
        if not python:
            self._cache.pop("fennec", None)
            return None, (f"no interpreter here can import {WEB_TRANSPORT}: install it, or "
                          f"point {WEB_PYTHON_ENV} at one that has it")
        if not self._decision_ready():
            self._cache.pop("fennec", None)
            return None, "LIQUID_API_KEY is not set, so d1.2 cannot decide an action"
        command, cwd, env = self._fennec_command(python, url, goal, max_steps, timeout)
        stdout, stderr, outcome = self._run_driver(command, cwd, env,
                                                   max(5, float(timeout)) + LANE_GRACE_S,
                                                   progress)
        if outcome == "timeout":
            self._cache.pop("fennec", None)
            return {"ok": False, "engine": "liquid-d1.2", "outlet": "fennec-webagent",
                    "error": f"the Fennec web loop did not finish inside {timeout}s"}, ""
        if outcome == "failed-to-start":
            return None, f"the Fennec web loop could not start: {stderr}"
        report = _report(stdout, WEB_SENTINEL)
        if report is None:
            return {"ok": False, "engine": "liquid-d1.2", "outlet": "fennec-webagent",
                    "error": redact((stderr or stdout or "").strip()[-2000:], self.env),
                    "returncode": outcome}, ""
        if outcome == 2:
            # The loop would not load in an interpreter that passed the transport probe
            # minutes ago. Nothing on the page has changed, so the answer is a clean "no".
            return None, str(report.get("error") or "the Fennec web loop would not load")
        report.setdefault("ok", outcome == 0)
        if outcome != 0 and report.get("ok"):
            report["ok"] = False
            report["error"] = f"the Fennec web driver exited {outcome} after reporting success"
        report["outlet"] = "fennec-webagent"
        return report, ""

    def _run_driver(self, command, cwd, env, timeout, progress=None):
        """Run a driver with files behind the pipes instead of pipes themselves.

        capture_output=True makes subprocess drain the pipes in its own threads, and on
        Windows a browser grandchild that inherits them keeps them open, so a wait that was
        supposed to end at the timeout keeps waiting for something that is not this process.
        Files have no such second holder: the report is read when the driver is gone.

        With a ``progress`` callback the driver is additionally polled rather than waited on,
        and the notes it appends to its own file are handed over as they land.
        """
        options = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
        notes = None
        if progress is not None:
            # Named, because the child opens it by name while the parent reads it by name;
            # it is created empty, so the tail starts at zero bytes rather than missing file.
            handle, notes = tempfile.mkstemp(prefix="fennec-notes-", suffix=".jsonl")
            os.close(handle)
            command = list(command) + ["--progress", notes]
        state = {"offset": 0, "rest": b""}
        drain = None if notes is None else (
            lambda: _tail_notes(notes, state, progress, env))
        try:
            with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
                try:
                    proc = subprocess.Popen(command, cwd=cwd, env=env,
                                            stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                                            **options)
                except OSError as exc:
                    return "", redact(str(exc), env), "failed-to-start"
                outcome = self._poll_driver(proc, time.monotonic() + max(0.05, float(timeout)),
                                            drain)
                out.seek(0)
                err.seek(0)
                return (out.read(WEB_REPORT_BYTES).decode("utf-8", "replace"),
                        err.read(WEB_ERROR_BYTES).decode("utf-8", "replace"), outcome)
        finally:
            if notes is not None:
                # The temp dir is not this file's garbage collector: a lane that left one
                # file per browse would fill it, and the browser already leaves enough there.
                try:
                    os.unlink(notes)
                except OSError:
                    pass

    @staticmethod
    def _poll_driver(proc, deadline, drain):
        """Wait for the driver, forwarding what it noted while it ran.

        Polling is what lets a step still inside a 25-second decision say so. The last look at
        the notes happens after the exit is noticed, because the line the loop writes as it
        finishes is the one that says how far it got, and it is already on disk by the time
        this process is seen to be gone.
        """
        while True:
            if drain is not None:
                drain()
            stopped = proc.poll()
            if stopped is not None:
                break
            if time.monotonic() >= deadline:
                proc.kill()
                try:
                    proc.wait(timeout=DRIVER_KILL_GRACE_S)
                except subprocess.TimeoutExpired:
                    pass
                stopped = "timeout"
                break
            time.sleep(DRIVER_POLL_S)
        if drain is not None:
            drain()
        return stopped

    def _fennec_command(self, python, url, goal, max_steps, timeout):
        driver = str(Path(__file__).with_name("web_driver.py"))
        # The loop is first-party, so it gets this whole environment: it needs the Liquid key
        # to decide an action and the engine credentials to word one. Proxy settings belong to
        # the caller's browser session; passing an authenticated proxy into the child makes
        # Firefox stop on a username/password prompt before the page can load.
        env = dict(self.env)
        for name in BROWSER_PROXY_ENV_KEYS:
            env.pop(name, None)
        for name in ("NO_PROXY", "no_proxy"):
            if env.get(name):
                env[name] = _bypass_list(env[name])
        path = [str(Path(__file__).resolve().parent)]
        if env.get("PYTHONPATH"):
            path.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(path)
        # A browser's own log lands on the child's stdout, and a page title with a "→" in
        # it would raise in the interpreter's default code page before the report is sent.
        env.setdefault("PYTHONIOENCODING", "utf-8")
        return ([python, driver, "--url", url, "--goal", goal,
                 "--max-steps", str(int(max_steps or 12)), "--timeout", str(int(timeout))],
                None, env)

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


def _report(raw, marker):
    """The driver's JSON, taken from after its sentinel instead of all of stdout.

    A driver runs in an interpreter of somebody else's choosing, with a browser attached,
    so a banner or a progress line on stdout would otherwise turn a finished, paid browser
    run into "the outlet printed no JSON". Anything on either side of the object is noise,
    so read one value and stop.
    """
    found = (raw or "").rfind(marker)
    if found < 0:
        return None
    try:
        value, _ = json.JSONDecoder().raw_decode(raw[found + len(marker):].lstrip())
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def _tail_notes(path, state, progress, env):
    """Forward the JSON lines the driver appended since the last look.

    A line is held back until its newline arrives: the writer puts one line down at a time,
    so the reader can land in the middle of one, and half a note is not worth showing. The
    control names inside these lines come from somebody else's website, so they pass through
    the same redaction as the rest of the transcript.
    """
    try:
        with open(path, "rb") as handle:
            handle.seek(state["offset"])
            chunk = handle.read(NOTES_BYTES)
    except OSError:
        return
    # The offset counts bytes taken out of the file, including the ones still held in ``rest``,
    # so the next read starts after them and only the text is replayed to the line they belong to.
    state["offset"] += len(chunk)
    lines = (state["rest"] + chunk).split(b"\n")
    state["rest"] = lines.pop()
    for raw in lines:
        if not raw.strip():
            continue
        note = _json_value(raw)
        if not isinstance(note, dict) or not note.get("text"):
            continue
        progress({"text": redact(str(note["text"]), env), "step": note.get("step"),
                  "of": note.get("of")})


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
