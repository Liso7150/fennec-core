"""The two optional web-agent outlets, driven without a browser and without money.

Jev is called through a subprocess we fake, OpenBot through a urllib opener we
fake. Nothing here reaches a paid API, launches a browser or touches either
upstream checkout, and no test can be misled by a checkout in the real home.
"""
from __future__ import annotations

import hermetic  # noqa: F401

import json
import subprocess
import sys
import tempfile
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from fennec_core import outlets as outlets_mod
from fennec_core.outlets import Outlets


class FakeResponse:
    """A urllib response: an iterable line stream that can also be read whole."""

    def __init__(self, lines):
        self._lines = lines

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        return iter(self._lines)

    def read(self, limit=None):
        raw = b"".join(self._lines)
        return raw if limit is None else raw[:limit]


def driver_stdout(payload):
    """What a Jev run leaves on stdout: the driver's own log, then the report."""
    return outlets_mod.JEV_SENTINEL + json.dumps(payload)


def fake_jev_process(stdout="", stderr="", returncode=0):
    """``subprocess.run`` with real files behind the pipes, as _run_jev asks for."""
    box = {}

    def run(command, **kwargs):
        box["command"] = command
        box.update({key: kwargs.get(key) for key in ("cwd", "env", "timeout", "stdin",
                                                     "creationflags")})
        for key, text in (("stdout", stdout), ("stderr", stderr)):
            stream = kwargs.get(key)
            if hasattr(stream, "write"):
                stream.write(text.encode("utf-8"))
        return SimpleNamespace(returncode=returncode)

    return box, mock.patch.object(outlets_mod.subprocess, "run", run)


def sse(events):
    return [b"data: " + json.dumps(event).encode() + b"\n" for event in events] + [b"data: [DONE]\n"]


def json_body(payload):
    return FakeResponse([json.dumps(payload).encode()])


class FakeOpener:
    """Answers the URLs the outlets probe, in the order the requests arrive."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def open(self, request, timeout=None):
        self.requests.append((request.full_url, getattr(request, "data", None), timeout))
        if not self.responses:
            raise urllib.error.URLError("nothing answering")
        value = self.responses.pop(0)
        if isinstance(value, Exception):
            raise value
        return value


def make_jev_root(base):
    root = base / "jev-ultrafast"
    (root / "jev_ultrafast").mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\nname = 'jev-ultrafast'\n", encoding="utf-8")
    return root


class OutletCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.home = self.root / "home"
        self.home.mkdir()
        patcher = mock.patch.object(Path, "home", lambda: self.home)
        patcher.start()
        self.addCleanup(patcher.stop)

    def outlets(self, environ=None, jev_root="", openbot_url="", suite_root=None):
        config = SimpleNamespace(jev_root=jev_root, openbot_url=openbot_url,
                                 suite_root=str(self.root) if suite_root is None else suite_root)
        return Outlets(config, environ or {})


class LoopbackTests(OutletCase):
    def test_only_http_loopback_counts_as_local(self):
        self.assertTrue(outlets_mod._loopback("http://127.0.0.1:3001"))
        self.assertTrue(outlets_mod._loopback("http://localhost:3001"))
        self.assertFalse(outlets_mod._loopback("https://127.0.0.1:3001"))
        self.assertFalse(outlets_mod._loopback("http://10.0.0.5:3001"))
        self.assertFalse(outlets_mod._loopback("not-a-url"))

    def test_non_loopback_openbot_is_refused_before_any_request(self):
        outlet = self.outlets({"FENNEC_OPENBOT_URL": "http://other-box:3001"})
        with mock.patch.object(urllib.request, "build_opener") as build:
            state = outlet.openbot_status(refresh=True)
            report = outlet.coworker_task("read the inbox")
        build.assert_not_called()
        self.assertFalse(state["present"])
        self.assertIn("machine token", state["error"])
        self.assertIn("machine token", report["error"])


class JevDiscoveryTests(OutletCase):
    def test_configured_root_wins_and_markers_are_required(self):
        wanted = make_jev_root(self.root / "elsewhere")
        self.assertEqual(self.outlets(jev_root=str(wanted)).jev_root(), wanted)
        (wanted / "pyproject.toml").unlink()
        self.assertIsNone(self.outlets(jev_root=str(wanted)).jev_root())

    def test_env_var_points_at_a_checkout_anywhere(self):
        wanted = make_jev_root(self.root / "somewhere")
        self.assertEqual(self.outlets({"FENNEC_JEV_ROOT": str(wanted)}).jev_root(), wanted)

    def test_suite_root_is_a_candidate(self):
        wanted = make_jev_root(self.root / "parent")
        self.assertEqual(self.outlets(suite_root=str(self.root / "parent")).jev_root(), wanted)

    def test_home_candidates_are_used_when_nothing_else_matches(self):
        self.assertIsNone(self.outlets(suite_root="").jev_root())
        relative = outlets_mod.DEFAULT_JEV_ROOTS[0]
        wanted = make_jev_root(self.home / Path(relative).parent)
        self.assertEqual(self.outlets(suite_root="").jev_root(), wanted)

    def test_a_suite_sibling_checkout_is_found_through_home_too(self):
        wanted = make_jev_root(self.home / "Fennec-Suite")
        self.assertEqual(self.outlets(suite_root="").jev_root(), wanted)

    def test_status_reports_the_policy_key_without_reading_its_value(self):
        wanted = make_jev_root(self.root)
        (wanted / ".env").write_text("TYPESAFE_API_KEY=super-secret\n", encoding="utf-8")
        status = self.outlets().jev_status(refresh=True)
        self.assertTrue(status["present"])
        self.assertEqual(status["root"], str(wanted))
        self.assertTrue(status["policy_key"])
        self.assertNotIn("super-secret", json.dumps(status))

    def test_env_policy_key_is_seen_too(self):
        make_jev_root(self.root)
        self.assertTrue(self.outlets({"TYPESAFE_API_KEY": "k"})
                        .jev_status(refresh=True)["policy_key"])

    def test_absent_jev_is_reported_as_absent(self):
        with mock.patch.object(Outlets, "jev_importable", return_value=False):
            outlet = self.outlets(suite_root="")
            self.assertFalse(outlet.jev_status(refresh=True)["present"])
            self.assertFalse(outlet.available("web_task"))


class JevCallTests(OutletCase):
    def setUp(self):
        super().setUp()
        self.jev = make_jev_root(self.root / "checkouts")

    def jev_outlet(self, environ=None):
        return self.outlets(environ, jev_root=str(self.jev))

    def capture_run(self, stdout=None):
        return fake_jev_process(stdout=driver_stdout({"ok": True, "steps": 3, "status": "done",
                                                      "url": "https://example.test/done"})
                                if stdout is None else stdout)

    def test_goal_and_end_state_reach_the_driver_unchanged(self):
        box, patch = self.capture_run()
        with mock.patch.object(outlets_mod.shutil, "which", return_value="/usr/bin/uv"), patch:
            report = self.jev_outlet().web_task("the cart shows two items",
                                                "https://shop.example.test/cart", max_steps=7)
        self.assertTrue(report["ok"], report)
        self.assertEqual(report["outlet"], "jev-ultrafast")
        self.assertEqual(box["command"][:2], ["uv", "run"])
        self.assertEqual(box["command"][box["command"].index("--project") + 1], str(self.jev))
        self.assertEqual(box["command"][-8:], ["--url", "https://shop.example.test/cart",
                                               "--goal", "the cart shows two items",
                                               "--max-steps", "7",
                                               "--timeout", str(int(outlets_mod.JEV_TIMEOUT_S))])
        self.assertEqual(box["cwd"], str(self.jev))
        self.assertEqual(box["env"].get("PYTHONPATH"), str(self.jev))
        self.assertEqual(box["timeout"], outlets_mod.JEV_TIMEOUT_S + 15)
        # A browser agent that inherits our stdin is a turn that never ends.
        self.assertIs(box["stdin"], subprocess.DEVNULL)

    def test_fennecs_own_keys_never_reach_the_browser_checkout(self):
        box, patch = self.capture_run()
        outlet = self.jev_outlet({"FENNEC_API_KEY": "fk-live", "COLI_API_KEY": "coli-live",
                                  "FENNEC_APP_TOKEN": "app-live", "TYPESAFE_API_KEY": "jev-live",
                                  "COMPUTERNAME": "KEEPME"})
        with mock.patch.object(outlets_mod.shutil, "which", return_value=None), patch:
            self.assertTrue(outlet.web_task("open")["ok"])
        self.assertNotIn("FENNEC_API_KEY", box["env"])
        self.assertNotIn("COLI_API_KEY", box["env"])
        self.assertNotIn("FENNEC_APP_TOKEN", box["env"])
        # Jev's own key belongs to Jev: it is upstream's, not ours to withhold.
        self.assertEqual(box["env"]["TYPESAFE_API_KEY"], "jev-live")
        self.assertEqual(box["env"]["COMPUTERNAME"], "KEEPME")

    def test_plain_python_is_used_when_uv_is_missing(self):
        box, patch = self.capture_run()
        with mock.patch.object(outlets_mod.shutil, "which", return_value=None), patch:
            self.assertTrue(self.jev_outlet().web_task("sign in")["ok"])
        self.assertEqual(box["command"][0], sys.executable)
        self.assertTrue(box["command"][1].endswith("jev_driver.py"))
        self.assertEqual(box["command"][2:4], ["--url", "https://www.google.com/"])

    def test_bad_requests_are_refused_before_a_process_starts(self):
        with mock.patch.object(outlets_mod.subprocess, "run") as run:
            self.assertIn("end state", self.jev_outlet().web_task("  ")["error"])
            self.assertIn("http(s)", self.jev_outlet().web_task("open", "ftp://x")["error"])
        run.assert_not_called()

    def test_upstream_failures_are_returned_not_raised(self):
        _, patch = fake_jev_process(stderr="browser exploded", returncode=1)
        with patch:
            report = self.jev_outlet().web_task("open the page")
        self.assertFalse(report["ok"])
        self.assertEqual(report["returncode"], 1)
        self.assertEqual(report["error"], "browser exploded")

    def test_an_upstream_banner_cannot_hide_the_report(self):
        noisy = ("booting chromium\n"
                 + driver_stdout({"ok": True, "steps": 2, "text": "done"})
                 + "\nbye\n")
        _, patch = fake_jev_process(stdout=noisy)
        with patch:
            report = self.jev_outlet().web_task("open the page")
        self.assertEqual(report["ok"], True)
        self.assertEqual(report["steps"], 2)

    def test_a_report_of_success_from_a_failed_process_is_not_believed(self):
        _, patch = fake_jev_process(stdout=driver_stdout({"ok": True, "steps": 9}), returncode=3)
        with patch:
            report = self.jev_outlet().web_task("open the page")
        self.assertFalse(report["ok"])
        self.assertIn("exited 3", report["error"])

    def test_a_leaked_key_in_an_upstream_traceback_is_not_passed_on(self):
        outlet = self.jev_outlet({"FENNEC_API_KEY": "fk-live"})
        _, patch = fake_jev_process(stderr="KeyError in Fennec: fk-live", returncode=1)
        with patch:
            report = outlet.web_task("open the page")
        self.assertNotIn("fk-live", report["error"])
        self.assertIn("[redacted]", report["error"])

    def test_metadata_addresses_are_refused_before_a_browser_starts(self):
        with mock.patch.object(outlets_mod.subprocess, "run") as run:
            for target in ("http://169.254.169.254/latest/meta-data/", "http://[::]/",
                           "http://224.0.0.1/"):
                report = self.jev_outlet().web_task("open", target)
                self.assertIn("link-local or reserved", report["error"], target)
        run.assert_not_called()

    def test_timeout_and_missing_interpreter_are_plain_errors(self):
        def explode(*a, **k):
            raise subprocess.TimeoutExpired(cmd="jev", timeout=1)
        with mock.patch.object(outlets_mod.subprocess, "run", explode):
            self.assertIn("timed out", self.jev_outlet().web_task("open")["error"])

        def die(*a, **k):
            raise OSError("no interpreter")
        with mock.patch.object(outlets_mod.subprocess, "run", die):
            self.assertIn("could not start", self.jev_outlet().web_task("open")["error"])

    def test_no_jev_checkout_says_how_to_get_one(self):
        report = self.outlets(jev_root=str(self.root / "nothing-here"),
                              suite_root="").web_task("open")
        self.assertIn("jev-ultrafast", report["error"])
        self.assertIn("github.com", report["error"])

    def test_available_follows_discovery(self):
        with mock.patch.object(Outlets, "jev_importable", return_value=False):
            self.assertTrue(self.jev_outlet().available("web_task"))
            self.assertFalse(self.outlets(suite_root="").available("web_task"))


class JevDriverTests(unittest.TestCase):
    """The driver only prints its report, and never imports upstream when it cannot."""

    def test_import_failure_is_a_json_object_behind_the_sentinel(self):
        import contextlib
        import io
        from fennec_core import jev_driver
        out = io.StringIO()
        with mock.patch.dict(sys.modules, {"jev_ultrafast": None}), \
             contextlib.redirect_stdout(out):
            code = jev_driver.main(["--url", "https://x.test", "--goal", "g"])
        self.assertEqual(code, 2)
        report = outlets_mod._report(out.getvalue())
        self.assertFalse(report["ok"])
        self.assertIn("not importable", report["error"])

    def test_the_sentinels_agree_between_driver_and_caller(self):
        from fennec_core import jev_driver
        self.assertEqual(jev_driver.SENTINEL, outlets_mod.JEV_SENTINEL)
        self.assertIsNone(outlets_mod._report("no report here"))


class OpenBotProbeTests(OutletCase):
    def test_agents_list_is_read_from_the_ag_ui_routes(self):
        opener = FakeOpener([json_body([{"id": "a1", "name": "Ivy"}, {"id": "a2", "name": "Otto"}])])
        outlet = self.outlets()
        with mock.patch.object(urllib.request, "build_opener", return_value=opener):
            state = outlet.openbot_status(refresh=True)
            self.assertTrue(opener.requests[0][0].endswith("/api/agents"))
            self.assertTrue(state["present"])
            self.assertEqual([a["name"] for a in state["agents"]], ["Ivy", "Otto"])
            self.assertTrue(outlet.available("coworker_task"))

    def test_auth_wall_explains_that_openbot_issues_no_machine_token(self):
        error = urllib.error.HTTPError("http://127.0.0.1:3001/api/agents", 401,
                                       "Unauthorized", None, None)
        with mock.patch.object(urllib.request, "build_opener", return_value=FakeOpener([error])):
            state = self.outlets().openbot_status(refresh=True)
        self.assertFalse(state["present"])
        self.assertIn("Better-Auth", state["error"])

    def test_configured_url_wins_over_the_default(self):
        self.assertEqual(self.outlets({"FENNEC_OPENBOT_URL": "http://127.0.0.1:4100/"})
                         .openbot_base(), "http://127.0.0.1:4100")
        self.assertEqual(self.outlets(openbot_url="http://localhost:3210").openbot_base(),
                         "http://localhost:3210")
        self.assertEqual(self.outlets().openbot_base(), outlets_mod.OPENBOT_DEFAULT)

    def test_status_is_cached_and_can_be_refreshed(self):
        opener = FakeOpener([json_body([]), json_body([])])
        outlet = self.outlets()
        with mock.patch.object(urllib.request, "build_opener", return_value=opener):
            outlet.openbot_status(refresh=True)
            outlet.openbot_status()
            self.assertEqual(len(opener.requests), 1)
            outlet.openbot_status(refresh=True)
            self.assertEqual(len(opener.requests), 2)

    def test_status_reports_both_outlets(self):
        make_jev_root(self.root)
        with mock.patch.object(urllib.request, "build_opener",
                               return_value=FakeOpener([json_body([{"id": "a1", "name": "Ivy"}])])):
            status = self.outlets().status(refresh=True)
        self.assertTrue(status["jev"]["present"])
        self.assertTrue(status["openbot"]["present"])
        self.assertEqual(status["openbot"]["url"], outlets_mod.OPENBOT_DEFAULT)


class OpenBotRunTests(OutletCase):
    def run_with(self, events, environ=None, **kwargs):
        opener = FakeOpener([json_body([{"id": "solo", "name": "Solo"}]), FakeResponse(events)])
        with mock.patch.object(urllib.request, "build_opener", return_value=opener):
            report = self.outlets(environ).coworker_task("summarise the thread", **kwargs)
        return report, opener

    def test_the_ag_ui_run_is_one_post_and_the_text_is_reassembled(self):
        report, opener = self.run_with(sse([{"type": "RUN_STARTED"},
                                            {"type": "TEXT_MESSAGE_CONTENT", "delta": "here is "},
                                            {"type": "TEXT_MESSAGE_CONTENT", "delta": "the answer"},
                                            {"type": "RUN_FINISHED"}]))
        self.assertTrue(report["ok"])
        self.assertEqual(report["outlet"], "openbot")
        self.assertEqual(report["answer"], "here is the answer")
        self.assertTrue(report["complete"])
        self.assertEqual(report["steps"], 4)
        url, data, timeout = opener.requests[1]
        self.assertEqual(url, "http://127.0.0.1:3001/api/copilotkit/agent/solo/run")
        body = json.loads(data.decode())
        self.assertEqual(body["messages"][0]["content"], "summarise the thread")
        self.assertEqual(body["messages"][0]["role"], "user")
        self.assertTrue(body["threadId"] and body["runId"])
        self.assertEqual(timeout, outlets_mod.OPENBOT_TIMEOUT_S)

    def test_named_agent_is_matched_by_id_or_name(self):
        opener = FakeOpener([json_body([{"id": "a1", "name": "Ivy"}, {"id": "a2", "name": "Otto"}]),
                             FakeResponse(sse([{"type": "RUN_FINISHED"}]))])
        with mock.patch.object(urllib.request, "build_opener", return_value=opener):
            self.outlets().coworker_task("do it", agent="Otto")
        self.assertIn("/agent/a2/run", opener.requests[1][0])

    def test_ambiguous_and_unknown_agents_ask_for_a_name(self):
        agents = [{"id": "a1", "name": "Ivy"}, {"id": "a2", "name": "Otto"}]
        opener = FakeOpener([json_body(agents)])
        with mock.patch.object(urllib.request, "build_opener", return_value=opener):
            report = self.outlets().coworker_task("do it")
        self.assertIn("more than one coworker", report["error"])
        self.assertEqual(report["agents"], agents)
        opener = FakeOpener([json_body(agents)])
        with mock.patch.object(urllib.request, "build_opener", return_value=opener):
            report = self.outlets().coworker_task("do it", agent="Ghost")
        self.assertIn("unknown OpenBot coworker", report["error"])

    def test_run_error_wins_over_the_text_it_reached(self):
        report, _ = self.run_with(sse([{"type": "TEXT_MESSAGE_CONTENT", "delta": "half a "},
                                       {"type": "RUN_ERROR", "message": "browser is busy"}]))
        self.assertFalse(report["ok"])
        self.assertEqual(report["error"], "browser is busy")

    def test_a_truncated_stream_is_flagged_as_incomplete(self):
        report, _ = self.run_with(sse([{"type": "TEXT_MESSAGE_CONTENT", "delta": "half a"}]))
        self.assertTrue(report["ok"])
        self.assertFalse(report["complete"])
        report, _ = self.run_with(sse([{"type": "RUN_STARTED"}]))
        self.assertIn("without RUN_FINISHED", report["error"])

    def test_thread_id_reuses_the_openbot_conversation(self):
        report, opener = self.run_with(sse([{"type": "RUN_FINISHED"}]), thread_id="t-77")
        self.assertEqual(json.loads(opener.requests[1][1].decode())["threadId"], "t-77")
        self.assertEqual(report["thread"], "t-77")

    def test_connection_failure_names_the_transport_not_a_stack_trace(self):
        opener = FakeOpener([urllib.error.URLError("down")])
        with mock.patch.object(urllib.request, "build_opener", return_value=opener):
            report = self.outlets().coworker_task("do it")
        self.assertFalse(report["ok"])
        self.assertIn("no OpenBot workspace", report["error"])

    def test_empty_goal_is_refused_without_a_request(self):
        with mock.patch.object(urllib.request, "build_opener") as build:
            report = self.outlets().coworker_task("   ")
        build.assert_not_called()
        self.assertIn("needs a goal", report["error"])

    def test_an_answering_workspace_with_no_agents_says_so(self):
        report = self.outlets()._openbot_agent_id([], "")
        self.assertIn("no coworker agent", report["error"])
        self.assertEqual(report["agents"], [])

    def test_a_coworker_without_a_usable_id_is_not_a_url(self):
        for listed in ([{"name": "Ivy"}], [{"id": "", "name": "Ivy"}], [{"id": {"x": 1}}]):
            report = self.outlets()._openbot_agent_id(listed, "")
            self.assertFalse(report["ok"], listed)
            self.assertIn("no usable id", report["error"])

    def test_an_agent_id_cannot_escape_the_run_path(self):
        report_id = self.outlets()._openbot_agent_id([{"id": "a/../../admin", "name": "Ivy"}], "")
        self.assertEqual(report_id, "a%2F..%2F..%2Fadmin")

    def test_a_run_that_outlives_its_event_budget_stops(self):
        stream = sse([{"type": "TEXT_MESSAGE_CONTENT", "delta": "x"}] * 4)
        opener = FakeOpener([json_body([{"id": "solo", "name": "Solo"}]), FakeResponse(stream)])
        with mock.patch.object(urllib.request, "build_opener", return_value=opener), \
             mock.patch.object(outlets_mod, "OPENBOT_EVENT_CAP", 3):
            report = self.outlets().coworker_task("chatter")
        self.assertFalse(report["ok"])
        self.assertIn("more than 3 events", report["error"])
        self.assertEqual(report["answer"], "xxx")

    def test_a_workspace_that_dies_mid_run_is_not_left_marked_present(self):
        outlet = self.outlets()
        opener = FakeOpener([json_body([{"id": "solo", "name": "Solo"}]),
                             urllib.error.URLError("gone")])
        with mock.patch.object(urllib.request, "build_opener", return_value=opener):
            self.assertTrue(outlet.openbot_status(refresh=True)["present"])
            report = outlet.coworker_task("do it")
        self.assertIn("stopped answering", report["error"])
        self.assertNotIn("openbot", outlet._cache)

    def test_the_openbot_opener_refuses_redirects(self):
        opener = outlets_mod._opener()
        self.assertTrue(any(isinstance(handler, outlets_mod._NoRedirects)
                            for handler in opener.handlers))
        moved = urllib.error.HTTPError("http://127.0.0.1:3001/api/agents", 302, "Found", None, None)
        with mock.patch.object(urllib.request, "build_opener", return_value=FakeOpener([moved])):
            state = self.outlets().openbot_status(refresh=True)
        self.assertFalse(state["present"])
        self.assertIn("HTTP 302", state["error"])

    def test_a_url_that_only_looks_loopback_is_not_loopback(self):
        self.assertFalse(outlets_mod._loopback("http://evil.com@127.0.0.1:3001"))
        outlet = self.outlets({"FENNEC_OPENBOT_URL": "http://evil.com@127.0.0.1:3001"})
        with mock.patch.object(urllib.request, "build_opener") as build:
            state = outlet.openbot_status(refresh=True)
        build.assert_not_called()
        self.assertFalse(state["present"])
        self.assertIn("machine token", state["error"])


class ToolBridgeTests(OutletCase):
    def test_tools_without_an_outlet_object_explain_themselves(self):
        from fennec_core import tools
        ctx = SimpleNamespace(outlets=None)
        self.assertIn("no web outlets", tools.web_task(ctx, "open a page")["error"])
        self.assertIn("no web outlets", tools.coworker_task(ctx, "open a page")["error"])

    def test_availability_is_a_hard_no_when_the_outlet_is_absent(self):
        from fennec_core import tools
        with mock.patch.object(Outlets, "jev_importable", return_value=False), \
             mock.patch.object(urllib.request, "build_opener", return_value=FakeOpener([])):
            ctx = SimpleNamespace(outlets=self.outlets(suite_root=""), allow_web=True,
                                  require_approval=False, extra={})
            self.assertFalse(tools._available("web_task", ctx))
            self.assertFalse(tools._available("coworker_task", ctx))

    def test_a_present_outlet_makes_the_tool_visible_and_web_gated(self):
        from fennec_core import tools
        make_jev_root(self.root)
        ctx = SimpleNamespace(outlets=self.outlets(), allow_web=True,
                              require_approval=False, extra={})
        self.assertTrue(tools._available("web_task", ctx))
        self.assertTrue(tools._permitted("web_task", ctx))
        ctx.allow_web = False
        self.assertFalse(tools._permitted("web_task", ctx))
        ctx.allow_web = True
        ctx.require_approval = True
        self.assertFalse(tools._permitted("web_task", ctx))

    def test_delegation_goes_through_the_session_outlet(self):
        from fennec_core import tools
        outlet = self.outlets()
        seen = []
        outlet.web_task = lambda goal, url, max_steps=12: seen.append(("jev", goal, url, max_steps))
        outlet.coworker_task = lambda goal, agent: seen.append(("openbot", goal, agent))
        ctx = SimpleNamespace(outlets=outlet)
        tools.web_task(ctx, "the invoice is on screen", "https://x.test", max_steps=4)
        tools.coworker_task(ctx, "file the report", "Ivy")
        self.assertEqual(seen, [("jev", "the invoice is on screen", "https://x.test", 4),
                                ("openbot", "file the report", "Ivy")])


if __name__ == "__main__":
    unittest.main()
