"""The optional web-agent outlets, driven without a browser and without money.

Fennec's own loop is called through a subprocess we fake, OpenBot through a urllib opener
we fake. Nothing here reaches a paid API, launches a browser or needs another project's
checkout, and no test can be misled by a venv in the real home.
"""
from __future__ import annotations

import hermetic  # noqa: F401

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import liquid
import outlets as outlets_mod
from outlets import Outlets


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


def driver_stdout(payload, marker=None):
    """What a driver run leaves on stdout: its own log noise, then the report.

    The marker is pinned against the driver module itself further down this file, so a
    report written with the caller's marker is the same object either way.
    """
    return (marker or outlets_mod.WEB_SENTINEL) + json.dumps(payload)


def web_stdout(payload):
    """The same, spelled for the report Fennec's own loop writes."""
    return driver_stdout(payload, outlets_mod.WEB_SENTINEL)


def fake_driver(stdout="", stderr="", returncode=0, notes=(), codes=None):
    """``subprocess.Popen`` for the driver lane, with real files behind the pipes.

    The fake writes into the capture files the lane opened and answers ``poll`` from
    ``codes``, appending one ``notes`` line per look it is taken - which is how a real driver
    spends its minutes: a line when a step starts, its exit only at the end. A caller that
    wants the wait itself patches ``_run_driver`` rather than reading a deadline off a clock.
    """
    box = {}
    if codes is None:
        # A driver that noted anything was alive while it did it, and the parent takes one look
        # per step, so the notes arrive on the polls before the exit rather than after it.
        codes = [None] * len(notes) + [returncode]
    left = list(codes)

    class Proc:
        def __init__(self, command, **kwargs):
            box["command"] = command
            box.update({key: kwargs.get(key) for key in ("cwd", "env", "stdin",
                                                         "creationflags")})
            box["starts"] = box.get("starts", 0) + 1
            for key, text in (("stdout", stdout), ("stderr", stderr)):
                stream = kwargs.get(key)
                if hasattr(stream, "write"):
                    stream.write(text.encode("utf-8"))
            self.path = command[command.index("--progress") + 1] if "--progress" in command else None
            self.notes = list(notes)
            self.left = left
            self.wrote = 0

        def poll(self):
            if len(self.left) > 1:
                if self.path and self.wrote < len(self.notes):
                    with open(self.path, "a", encoding="utf-8") as handle:
                        handle.write(json.dumps(self.notes[self.wrote]) + "\n")
                    self.wrote += 1
                return self.left.pop(0)
            return self.left[0]

        def kill(self):
            box["killed"] = True

        def wait(self, timeout=None):
            box["grace"] = timeout
            return 0

    return box, mock.patch.object(outlets_mod.subprocess, "Popen", Proc)


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


def browser_env(home):
    """The interpreter path Fennec's own browser venv would have under this fake home."""
    bins = ("Scripts", "python.exe") if os.name == "nt" else ("bin", "python")
    return home.joinpath(*outlets_mod.BROWSER_ENV, ".venv", *bins)


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

    def outlets(self, environ=None, openbot_url="", suite_root=None):
        config = SimpleNamespace(openbot_url=openbot_url,
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


class BrowserEnvTests(OutletCase):
    """Where the loop's transport comes from now that the lane belongs to Fennec."""

    def make_env(self):
        venv = browser_env(self.home)
        venv.parent.mkdir(parents=True)
        venv.write_text("", encoding="utf-8")
        return venv

    def test_a_machine_without_the_env_has_no_own_interpreter(self):
        self.assertEqual(self.outlets().browser_env_python(), "")

    def test_the_env_under_the_fennec_home_is_the_candidate(self):
        venv = self.make_env()
        self.assertEqual(self.outlets().browser_env_python(), str(venv))

    def test_candidates_are_the_pin_then_this_interpreter_then_the_env(self):
        venv = self.make_env()
        outlet = self.outlets({"FENNEC_WEB_PYTHON": "/opt/py"})
        self.assertEqual(outlet.web_candidates(), ["/opt/py", sys.executable, str(venv)])
        # A pin that is this interpreter is one candidate, not two, and no pin at all
        # still leaves the env last: it is the one the browser-outlets feature fills.
        pinned = self.outlets({"FENNEC_WEB_PYTHON": sys.executable})
        self.assertEqual(pinned.web_candidates(), [sys.executable, str(venv)])
        self.assertEqual(self.outlets().web_candidates(), [sys.executable, str(venv)])

    def test_the_status_names_where_the_browser_transport_came_from(self):
        venv = self.make_env()
        outlet = self.outlets()
        self.assertEqual(outlet.web_transport(""), "none")
        self.assertEqual(outlet.web_transport(sys.executable), "this interpreter")
        self.assertEqual(outlet.web_transport(str(venv)), "the Fennec browser env")
        self.assertEqual(outlet.web_transport(str(self.root / "elsewhere")), "external")
        pinned = self.outlets({"FENNEC_WEB_PYTHON": "/opt/py"})
        self.assertEqual(pinned.web_transport("/opt/py"), "pinned")
        # A pin that lost is still reported by where the answer actually came from.
        self.assertEqual(pinned.web_transport(str(venv)), "the Fennec browser env")
        with mock.patch.object(Outlets, "web_interpreter", return_value=str(venv)):
            self.assertEqual(pinned.fennec_status(refresh=True)["transport_from"],
                             "the Fennec browser env")


class FennecWebTests(OutletCase):
    """Fennec's own loop as the only browser outlet, and every way it can decline.

    There is no second agent behind these assertions. A lane that cannot start says so and
    leaves the page alone; the alternative was handing the user's browser to an agent they
    never chose after the one they did had already clicked on the page.
    """

    def loop_outlet(self, environ=None):
        return self.outlets({"LIQUID_API_KEY": "liquid-live", **(environ or {})})

    def capable(self, python="/capable/python"):
        return mock.patch.object(Outlets, "web_interpreter", return_value=python)

    def test_the_loop_answers_with_the_report_it_printed(self):
        box, patch = fake_driver(
            stdout=web_stdout({"ok": True, "engine": "liquid-d1.2", "steps": 3,
                               "status": "done", "url": "https://example.test/done"}))
        with self.capable(), patch:
            report = self.loop_outlet().web_task("the cart shows two items",
                                                 "https://shop.example.test/cart", max_steps=7)
        self.assertTrue(report["ok"], report)
        self.assertEqual(report["engine"], "liquid-d1.2")
        self.assertEqual(report["outlet"], "fennec-webagent")
        command = box["command"]
        self.assertEqual(command[0], "/capable/python")
        self.assertTrue(command[1].endswith("web_driver.py"))
        self.assertEqual(command[-8:], ["--url", "https://shop.example.test/cart",
                                        "--goal", "the cart shows two items",
                                        "--max-steps", "7",
                                        "--timeout", str(int(outlets_mod.WEB_TIMEOUT_S))])
        self.assertIsNone(box["cwd"])
        self.assertIs(box["stdin"], subprocess.DEVNULL)

    def test_the_lane_waits_for_the_request_plus_the_grace_to_exit(self):
        """Killing the driver at its own deadline would throw away the report it prints on the way out."""
        seen = []

        def record(self_, command, cwd, env, timeout, progress=None):
            seen.append(timeout)
            return web_stdout({"ok": True}), "", 0

        with self.capable(), mock.patch.object(Outlets, "_run_driver", record):
            outlet = self.loop_outlet()
            outlet.web_task("open the page")
            outlet.web_task("open the page", timeout=120)
        self.assertEqual(seen, [outlets_mod.WEB_TIMEOUT_S + outlets_mod.LANE_GRACE_S,
                                120.0 + outlets_mod.LANE_GRACE_S])

    def test_the_child_is_told_where_this_checkout_is_and_how_to_print(self):
        box, patch = fake_driver(stdout=web_stdout({"ok": True}))
        with self.capable(), patch:
            self.loop_outlet().web_task("open the page")
        env = box["env"]
        here = str(Path(outlets_mod.__file__).resolve().parent)
        self.assertEqual(env["PYTHONPATH"], here)
        self.assertEqual(env["PYTHONIOENCODING"], "utf-8")
        # The loop is first-party: it decides with d1.2 and words a field with the engine, so
        # it gets this whole environment rather than a scrubbed one.
        self.assertEqual(env["LIQUID_API_KEY"], "liquid-live")

    def test_an_existing_search_path_is_kept_behind_this_checkout(self):
        box, patch = fake_driver(stdout=web_stdout({"ok": True}))
        with self.capable(), patch:
            self.loop_outlet({"PYTHONPATH": "/somewhere/else"}).web_task("open the page")
        self.assertEqual(box["env"]["PYTHONPATH"].split(os.pathsep),
                         [str(Path(outlets_mod.__file__).resolve().parent), "/somewhere/else"])

    def test_the_proxy_bypass_reaches_the_child_in_the_form_httpx_reads(self):
        """The bypass is respelled, not dropped: the transport builds its client as it imports."""
        box, patch = fake_driver(stdout=web_stdout({"ok": True}))
        with self.capable(), patch:
            self.loop_outlet({"NO_PROXY": "[::1], localhost ,::1",
                              "no_proxy": "127.0.0.1,[fe80::1%eth0]"}).web_task("open the page")
        self.assertEqual(box["env"]["NO_PROXY"], "::1,localhost")
        self.assertEqual(box["env"]["no_proxy"], "127.0.0.1,fe80::1%eth0")
        plain, patch = fake_driver(stdout=web_stdout({"ok": True}))
        with self.capable(), patch:
            self.loop_outlet().web_task("open the page")
        self.assertNotIn("NO_PROXY", plain["env"], "no bypass is invented for a machine that has none")

    def test_no_key_declines_without_starting_a_process(self):
        with self.capable(), mock.patch.object(outlets_mod.subprocess, "Popen") as spawn:
            report = self.outlets().web_task("open the page")
        self.assertFalse(report["ok"])
        self.assertEqual(report["engine"], "liquid-d1.2")
        self.assertEqual(report["outlet"], "fennec-webagent")
        self.assertIn("LIQUID_API_KEY", report["error"])
        spawn.assert_not_called()

    def test_no_transport_names_the_import_and_the_switch(self):
        with self.capable(""), mock.patch.object(outlets_mod.subprocess, "Popen") as spawn:
            report = self.loop_outlet().web_task("open the page")
        self.assertFalse(report["ok"])
        self.assertIn(outlets_mod.WEB_TRANSPORT, report["error"])
        self.assertIn(outlets_mod.WEB_PYTHON_ENV, report["error"])
        spawn.assert_not_called()

    def test_a_page_the_lane_must_not_open_is_refused_before_any_process(self):
        with self.capable(), mock.patch.object(outlets_mod.subprocess, "Popen") as spawn:
            outlet = self.loop_outlet()
            reports = [outlet.web_task(""),
                       outlet.web_task("   "),
                       outlet.web_task("open", "ftp://example.test/file"),
                       outlet.web_task("open", "http://169.254.169.254/latest/meta-data/"),
                       outlet.web_task("open", "http://[::1/"),
                       outlet.web_task("open", "http://0.0.0.0:8184/")]
        spawn.assert_not_called()
        self.assertIn("goal", reports[0]["error"])
        self.assertIn("goal", reports[1]["error"])
        self.assertIn("http(s)", reports[2]["error"])
        self.assertIn("link-local or reserved", reports[3]["error"])
        self.assertIn("not a parseable", reports[4]["error"])
        self.assertIn("link-local or reserved", reports[5]["error"])
        for report in reports:
            self.assertFalse(report["ok"])
            self.assertNotIn("outlet", report, "refused before any lane was chosen")

    def test_a_loop_that_would_not_load_declines_cleanly(self):
        """Exit 2 means the driver never opened a tab, so the honest answer is a plain no."""
        box, patch = fake_driver(
            stdout=web_stdout({"ok": False, "engine": "liquid-d1.2",
                               "error": "webagent is not importable: x"}), returncode=2)
        with self.capable(), patch:
            report = self.loop_outlet().web_task("open the page")
        self.assertFalse(report["ok"])
        self.assertIn("not importable", report["error"])
        self.assertEqual(report["outlet"], "fennec-webagent")

    def test_a_loop_that_ran_out_of_time_is_not_given_to_anybody_else(self):
        box, patch = fake_driver()
        with self.capable(), patch, \
                mock.patch.object(Outlets, "_poll_driver", return_value="timeout"):
            report = self.loop_outlet().web_task("open the page")
        self.assertIn("did not finish", report["error"])
        self.assertEqual(report["outlet"], "fennec-webagent")
        self.assertEqual(box["starts"], 1, "a page the loop already clicked on is not a retry")
        self.assertTrue(box["command"][1].endswith("web_driver.py"))

    def test_a_driver_that_overruns_is_killed_rather_than_watched(self):
        """The wait is a deadline, not a promise the child will keep.

        A driver wedged by a browser grandchild never reports an exit, and the lane that
        waits for one holds the chat open past the turn. The grace period on the kill is
        what keeps a child that ignores SIGTERM from outliving its own answer.
        """
        class NeverExits:
            def __init__(self):
                self.killed = False

            def poll(self):
                return None

            def kill(self):
                self.killed = True

            def wait(self, timeout=None):
                self.grace = timeout
                return 0

        proc = NeverExits()
        outcome = Outlets._poll_driver(proc, time.monotonic() - 1.0, None)
        self.assertEqual(outcome, "timeout")
        self.assertTrue(proc.killed, "the wait ended because the driver was stopped, not because it answered")
        self.assertEqual(proc.grace, outlets_mod.DRIVER_KILL_GRACE_S)

    def test_a_note_written_as_the_driver_leaves_is_still_forwarded(self):
        """The line that says how far the loop got is written last and read last.

        A driver appends its note and exits in the same breath, so a wait that returns the
        moment it sees the exit forwards nothing and the chat loses the one step it had not
        shown. Both ways out - the exit and the kill - take one more look at the file.
        """
        path = self.root / "leaving.jsonl"
        note = {"text": "Deciding step 4 of 4", "step": 4, "of": 4}
        state = {"offset": 0, "rest": b""}
        seen = []

        def drain():
            outlets_mod._tail_notes(str(path), state, seen.append, {})

        class LeavesAtOnce:
            def poll(self):
                with path.open("a", encoding="utf-8") as notes:
                    notes.write(json.dumps(note) + "\n")
                return 0

        self.assertEqual(Outlets._poll_driver(LeavesAtOnce(), time.monotonic() + 60.0, drain), 0)
        self.assertEqual([line["text"] for line in seen], ["Deciding step 4 of 4"],
                         "a note that lands with the exit is not lost with the exit")

        path.unlink()
        state["offset"] = 0
        seen.clear()

        class LeavesWhenKilled:
            def poll(self):
                return None

            def kill(self):
                with path.open("a", encoding="utf-8") as notes:
                    notes.write(json.dumps(note) + "\n")

            def wait(self, timeout=None):
                return 0

        stopped = Outlets._poll_driver(LeavesWhenKilled(), time.monotonic() - 1.0, drain)
        self.assertEqual(stopped, "timeout")
        self.assertEqual([line["text"] for line in seen], ["Deciding step 4 of 4"],
                         "a driver that ran out of time still says where it stopped")

    def test_a_loop_that_could_not_be_launched_is_a_refusal_not_a_retry(self):
        seen = []

        def spawn(command, **kwargs):
            seen.append(command[1])
            raise OSError("interpreter vanished")

        with self.capable(), mock.patch.object(outlets_mod.subprocess, "Popen", spawn):
            report = self.loop_outlet().web_task("open the page")
        self.assertFalse(report["ok"])
        self.assertIn("could not start", report["error"])
        self.assertEqual(report["outlet"], "fennec-webagent")
        self.assertEqual(len(seen), 1, "one lane, refused, no second agent")

    def test_a_report_of_success_from_a_failed_loop_is_not_believed(self):
        box, patch = fake_driver(stdout=web_stdout({"ok": True, "steps": 9}), returncode=1)
        with self.capable(), patch:
            report = self.loop_outlet().web_task("open the page")
        self.assertFalse(report["ok"])
        self.assertIn("exited 1", report["error"])

    def test_a_browser_log_on_stdout_cannot_hide_the_report(self):
        """The caller reads behind the marker, so Chrome's own noise costs nothing."""
        box, patch = fake_driver(stdout="chrome: devtools listening on :9222\n"
                                         + web_stdout({"ok": True, "steps": 2}) + "\nlate log")
        with self.capable(), patch:
            report = self.loop_outlet().web_task("open the page")
        self.assertTrue(report["ok"], report)
        self.assertEqual(report["steps"], 2)

    def test_a_secret_in_the_child_output_never_reaches_the_transcript(self):
        box, patch = fake_driver(stdout="Traceback: used fennec-secret",
                                 stderr="authorization: fennec-secret", returncode=1)
        with self.capable(), patch:
            report = self.loop_outlet({"FENNEC_API_KEY": "fennec-secret"}).web_task("open the page")
        self.assertFalse(report["ok"])
        self.assertNotIn("fennec-secret", json.dumps(report))
        self.assertIn("[redacted]", report["error"])

    def test_this_interpreter_is_asked_in_process_and_never_spawned(self):
        with mock.patch.object(outlets_mod.importlib.util, "find_spec",
                               return_value=object()) as find, \
             mock.patch.object(outlets_mod.subprocess, "Popen") as spawn:
            self.assertTrue(self.outlets()._can_browse(sys.executable))
            self.assertFalse(self.outlets()._can_browse(""))
        find.assert_called_once_with(outlets_mod.WEB_TRANSPORT)
        spawn.assert_not_called()

    def test_a_broken_transport_import_is_a_no_not_an_error(self):
        def boom(name):
            raise ModuleNotFoundError(f"No module named {name!r}")

        with mock.patch.object(outlets_mod.importlib.util, "find_spec", boom):
            self.assertFalse(self.outlets()._can_browse(sys.executable))

    def test_another_interpreter_is_asked_once_and_the_answer_is_kept(self):
        other = self.root / "other-python"
        other.write_text("")
        seen = []

        def record(self_, command, cwd, env, timeout, progress=None):
            seen.append((command, timeout))
            return "", "", 0

        outlet = self.outlets()
        with mock.patch.object(Outlets, "_run_driver", record):
            self.assertTrue(outlet._can_browse(str(other)))
            self.assertTrue(outlet._can_browse(str(other)))
        self.assertEqual(len(seen), 1, "a status probe that spawns an interpreter per "
                                       "refresh costs more than the browser it reports on")
        command, timeout = seen[0]
        self.assertEqual(command, [str(other), "-c", f"import {outlets_mod.WEB_TRANSPORT}"])
        self.assertEqual(timeout, outlets_mod.BROWSE_PROBE_S)
        self.assertEqual(list(outlet._cache), [f"browse:{other}"])

    def test_a_probe_that_did_not_run_is_a_no(self):
        other = self.root / "other-python"
        other.write_text("")
        _, patch = fake_driver(returncode=1)
        with patch:
            self.assertFalse(self.outlets()._can_browse(str(other)))

    def test_the_same_interpreter_written_two_ways_is_one_interpreter(self):
        here = self.root / "python.exe"
        here.write_text("")
        # os.path.join keeps the "." segment a PurePath would drop, so this really is a
        # second spelling of the same file rather than the same string twice.
        self.assertTrue(outlets_mod._same_file(str(here),
                                               os.path.join(str(self.root), ".", here.name)))
        self.assertTrue(outlets_mod._same_file(sys.executable, sys.executable))
        self.assertFalse(outlets_mod._same_file(str(here), str(self.root / "other-python")))
        self.assertFalse(outlets_mod._same_file(str(here), ""))
        self.assertFalse(outlets_mod._same_file("", ""))

    def test_status_reports_the_loop_and_never_prints_the_key(self):
        with self.capable("/capable/python"):
            state = self.loop_outlet().fennec_status(refresh=True)
        self.assertTrue(state["present"])
        self.assertEqual(state["interpreter"], "/capable/python")
        self.assertTrue(state["decision_key"])
        self.assertEqual(state["model"], liquid.DEFAULT_MODEL)
        self.assertNotIn("liquid-live", json.dumps(state))
        with self.capable(""):
            self.assertFalse(self.loop_outlet().fennec_status(refresh=True)["present"])

    def test_the_status_block_carries_the_outlets_this_suite_ships(self):
        with self.capable(""), mock.patch.object(
                urllib.request, "build_opener", return_value=FakeOpener([json_body([])])):
            status = self.outlets().status(refresh=True)
        self.assertEqual(sorted(status), ["fennec", "openbot"])
        self.assertFalse(status["fennec"]["present"])

    def test_web_task_is_advertised_only_when_both_halves_stand(self):
        """An interpreter with no key and a key with no interpreter are both no browser tool."""
        with self.capable(""), mock.patch.object(urllib.request, "build_opener",
                                                 return_value=FakeOpener([])):
            self.assertFalse(self.outlets().available("web_task"))
        with self.capable("/capable/python"):
            self.assertFalse(self.outlets().available("web_task"), "no d1.2 key, no tool")
            self.assertTrue(self.loop_outlet().available("web_task"))
        self.assertFalse(self.outlets().available("some_other_tool"))

    def test_the_gate_and_the_call_agree_because_there_is_one_lane(self):
        """/api/outlets/run asks the same question the loop advertises by, and gets one answer."""
        with self.capable(""), mock.patch.object(outlets_mod.subprocess, "Popen") as spawn:
            outlet = self.loop_outlet()
            self.assertFalse(outlet.available("web_task"))
            report = outlet.web_task("open the page")
        spawn.assert_not_called()
        self.assertFalse(report["ok"])

    # --- live notes --------------------------------------------------------
    def test_a_note_written_while_the_loop_works_reaches_the_caller(self):
        """The whole point: a chat that moves while a decision is still running."""
        box, patch = fake_driver(
            stdout=web_stdout({"ok": True, "steps": 2}),
            notes=[{"text": "Opening the page", "step": 1, "of": 4},
                   {"text": "Deciding step 1 of 4", "step": 1, "of": 4}])
        seen = []
        with self.capable(), patch:
            report = self.loop_outlet().web_task("open the page", progress=seen.append)
        self.assertTrue(report["ok"], report)
        self.assertEqual([note["text"] for note in seen],
                         ["Opening the page", "Deciding step 1 of 4"])
        self.assertEqual(seen[1]["step"], 1)
        self.assertEqual(seen[1]["of"], 4)
        self.assertEqual(box["command"][-2], "--progress")

    def test_notes_carry_the_page_control_names_through_the_same_redaction(self):
        """A label is somebody else's text: a key typed into a field must not ride along."""
        box, patch = fake_driver(
            stdout=web_stdout({"ok": True}),
            notes=[{"text": "Typing in card number fennec-secret", "step": 1, "of": 2}])
        seen = []
        with self.capable(), patch:
            self.loop_outlet({"FENNEC_API_KEY": "fennec-secret"}).web_task("open the page",
                                                                           progress=seen.append)
        self.assertEqual(len(seen), 1)
        self.assertNotIn("fennec-secret", seen[0]["text"])
        self.assertIn("Typing in card number", seen[0]["text"])

    def test_the_notes_file_is_left_behind_by_neither_a_run_nor_a_failure(self):
        box, patch = fake_driver(stdout=web_stdout({"ok": True}),
                                 notes=[{"text": "Opening the page", "step": 1, "of": 2}])
        with self.capable(), patch:
            self.loop_outlet().web_task("open the page", progress=lambda note: None)
        path = box["command"][-1]
        self.assertFalse(os.path.exists(path), "one temp file per browse is a leak the browser did not need")

    def test_a_lane_with_nobody_listening_does_not_name_a_notes_file(self):
        """The probe interpreter has no chat waiting on it, so it gets no second channel."""
        box, patch = fake_driver(stdout=web_stdout({"ok": True}))
        with self.capable(), patch:
            self.loop_outlet().web_task("open the page")
        self.assertNotIn("--progress", box["command"])

    def test_a_note_the_reader_catches_half_written_is_waited_for(self):
        """The writer flushes per line and the reader polls: a clipped line is worse than a late one.

        It also pins the offset. A reader that rewinds over the bytes it already took out of
        the file re-reads them on every subsequent look, so the measure of correctness is that
        a caught-up reader sits at the end of the file with nothing held back.
        """
        path = self.root / "notes.jsonl"
        state = {"offset": 0, "rest": b""}
        seen = []

        def tail():
            outlets_mod._tail_notes(str(path), state, seen.append, {})

        def write(text):
            with path.open("a", encoding="utf-8") as notes:
                notes.write(text)

        write(json.dumps({"text": "Opening the page", "step": 1, "of": 3}) + "\n")
        tail()
        write(json.dumps({"text": "Deciding step 1 of 3", "step": 1, "of": 3}))
        tail()
        self.assertEqual([note["text"] for note in seen], ["Opening the page"],
                         "half a note is not a note")
        write("\n" + json.dumps({"text": "Clicking Sign in", "step": 2, "of": 3}) + "\n")
        tail()
        self.assertEqual([note["text"] for note in seen],
                         ["Opening the page", "Deciding step 1 of 3", "Clicking Sign in"])
        self.assertEqual(state["offset"], path.stat().st_size,
                         "the bytes held in the partial line were read, so they count")
        tail()
        self.assertEqual(len(seen), 3, "a finished file says nothing new on the next look")
        write("{not json\n\n" + json.dumps({"text": "Typing in the search box"}) + "\n")
        tail()
        self.assertEqual([note["text"] for note in seen[3:]], ["Typing in the search box"],
                         "a broken line is skipped, not shown")
        self.assertEqual(state["offset"], path.stat().st_size)

    def test_a_note_that_names_nothing_is_not_forwarded(self):
        state = {"offset": 0, "rest": b""}
        seen = []
        path = self.root / "quiet.jsonl"
        path.write_text(json.dumps({"step": 2}) + "\n" + json.dumps("just a string") + "\n"
                        + json.dumps({"text": "Waiting for the page to update"}) + "\n",
                        encoding="utf-8")
        outlets_mod._tail_notes(str(path), state, seen.append, {})
        self.assertEqual([note["text"] for note in seen], ["Waiting for the page to update"])
        self.assertIsNone(seen[0]["step"], "a note with no meter arrives with nothing in that column")


class DriverContractTests(unittest.TestCase):
    """The driver and its caller must agree on the marker, or an answer is read as noise."""

    def test_the_sentinel_agrees_between_the_driver_and_the_caller(self):
        import web_driver
        self.assertEqual(web_driver.SENTINEL, outlets_mod.WEB_SENTINEL)
        self.assertIsNone(outlets_mod._report("no report here", web_driver.SENTINEL))

    def test_a_report_is_read_with_its_own_marker(self):
        raw = driver_stdout({"ok": True, "steps": 2}, "@@@SOME-OTHER-REPORT@@@")
        self.assertIsNotNone(outlets_mod._report(raw, "@@@SOME-OTHER-REPORT@@@"))
        self.assertIsNone(outlets_mod._report(raw, outlets_mod.WEB_SENTINEL))

    def test_only_a_json_object_after_the_marker_counts_as_a_report(self):
        marker = outlets_mod.WEB_SENTINEL
        self.assertIsNone(outlets_mod._report("", marker))
        self.assertIsNone(outlets_mod._report(driver_stdout("the page loaded", marker), marker))
        self.assertIsNone(outlets_mod._report(driver_stdout([{"ok": True}], marker), marker))
        self.assertEqual(outlets_mod._report("banner noise " + driver_stdout({"ok": True}, marker),
                                             marker), {"ok": True})
        # raw_decode stops at the end of the object, so a log line after it is noise, not an error.
        self.assertEqual(outlets_mod._report(driver_stdout({"ok": True}, marker) + "late log",
                                             marker), {"ok": True})


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
        with mock.patch.object(Outlets, "web_interpreter", return_value="/capable/python"), \
             mock.patch.object(urllib.request, "build_opener",
                               return_value=FakeOpener([json_body([{"id": "a1", "name": "Ivy"}])])):
            status = self.outlets({"LIQUID_API_KEY": "liquid-live"}).status(refresh=True)
        self.assertEqual(sorted(status), ["fennec", "openbot"])
        self.assertTrue(status["fennec"]["present"])
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
    def context(self, **kwargs):
        """A call context, with the one field the tool bridge reads always present.

        ``progress`` is a real part of the contract - the loop sets it around every call - so
        these fixtures carry it rather than the handler reaching for a missing attribute.
        """
        kwargs.setdefault("progress", None)
        return SimpleNamespace(**kwargs)

    def test_tools_without_an_outlet_object_explain_themselves(self):
        import tools
        ctx = self.context(outlets=None)
        self.assertIn("no web outlets", tools.web_task(ctx, "open a page")["error"])
        self.assertIn("no web outlets", tools.coworker_task(ctx, "open a page")["error"])

    def test_availability_is_a_hard_no_when_the_outlet_is_absent(self):
        import tools
        with mock.patch.object(Outlets, "web_interpreter", return_value=""), \
             mock.patch.object(urllib.request, "build_opener", return_value=FakeOpener([])):
            ctx = self.context(outlets=self.outlets(suite_root=""), allow_web=True,
                               require_approval=False, extra={})
            self.assertFalse(tools._available("web_task", ctx))
            self.assertFalse(tools._available("coworker_task", ctx))

    def test_a_present_outlet_makes_the_tool_visible_and_web_gated(self):
        import tools
        with mock.patch.object(Outlets, "web_interpreter", return_value="/capable/python"):
            ctx = self.context(outlets=self.outlets({"LIQUID_API_KEY": "liquid-live"}),
                               allow_web=True, require_approval=False, extra={})
            self.assertTrue(tools._available("web_task", ctx))
        self.assertTrue(tools._permitted("web_task", ctx))
        ctx.allow_web = False
        self.assertFalse(tools._permitted("web_task", ctx))
        ctx.allow_web = True
        ctx.require_approval = True
        self.assertFalse(tools._permitted("web_task", ctx))

    def test_a_lane_with_nothing_to_run_hides_the_tool(self):
        """A d1.2 key with no browser transport behind it must not tempt the model with a browse."""
        import tools
        with mock.patch.object(Outlets, "web_interpreter", return_value=""):
            ctx = self.context(outlets=self.outlets({"LIQUID_API_KEY": "liquid-live"}),
                               allow_web=True, require_approval=False, extra={})
            self.assertFalse(tools._available("web_task", ctx))

    def test_delegation_goes_through_the_session_outlet(self):
        import tools
        outlet = self.outlets()
        seen = []
        outlet.web_task = lambda goal, url, max_steps=12, progress=None: seen.append(
            ("fennec", goal, url, max_steps))
        outlet.coworker_task = lambda goal, agent: seen.append(("openbot", goal, agent))
        ctx = self.context(outlets=outlet)
        tools.web_task(ctx, "the invoice is on screen", "https://x.test", max_steps=4)
        tools.coworker_task(ctx, "file the report", "Ivy")
        self.assertEqual(seen, [("fennec", "the invoice is on screen", "https://x.test", 4),
                                ("openbot", "file the report", "Ivy")])

    def test_the_live_notes_of_a_call_reach_the_lane_that_made_it(self):
        """The loop hands the tool its own reporter; the browser lane is the only thing that uses it."""
        import tools
        outlet = self.outlets()
        seen = []
        outlet.web_task = lambda goal, url, max_steps=12, progress=None: seen.append(progress)
        report = tools.web_task(self.context(outlets=outlet, progress="the reporter"),
                                "open the page", "https://x.test")
        self.assertEqual(seen, ["the reporter"])


if __name__ == "__main__":
    unittest.main()
