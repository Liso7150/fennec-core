"""Fennec's browser loop, driven without a browser and without a billable call.

Two seams carry all of it. ``FakeBrowser`` implements the four methods the loop calls
and replays scripted page states, so the loop's decisions and accounting are observable
without Chrome. ``FakeCDP`` answers the CDP methods the transport calls, including what
the page-side JavaScript returned, so the hit-testing and input rules are covered too.
:mod:`liquid` is exercised for real through a scripted opener: the questions
built here are the ones that would be sent to d1.2, and the answers are ones d1.2's
documented response shape allows.

Nothing here reaches api.liquid.ai, spends money, or starts a browser.
"""
from __future__ import annotations

import hermetic  # noqa: F401

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import liquid, web_driver, webagent


def action(id, node, kind, label, *, role="button", value="", rect=None, **extra):
    action_dict = {"id": id, "node": node, "role": role, "label": label, "kind": kind,
                   "value": value, "rect": rect or {"x": 10, "y": 10, "w": 100, "h": 20}}
    action_dict.update(extra)
    return action_dict


def appended(id, kind, label, **extra):
    """``scroll_down``, ``scroll_up`` and ``wait`` in the shape a real page returns them.

    A live capture of the page reader in Chrome proved these arrive with only an id, a
    kind and a label (plus a scroll's delta): no node, no role, no value, no rect. The
    element-only helpers must therefore never be reached with one, which is why they are
    a separate constructor rather than an ``action()`` call with a spare node number.
    """
    return {"id": id, "kind": kind, "label": label, **extra}


def page(*, url="https://shop.test/search", title="Search", text="Search the catalogue",
         actions=(), scroll=None, marker=None, page_key=None, guards=None):
    """A page in exactly the shape the page reader returns.

    The page reader numbers the controls it found and then appends scroll and wait with
    their own literal ids, so the reader never renumbers a control the way a test might.
    """
    state = {"url": url, "title": title, "text": text, "w": 1120, "h": 780,
             "scroll": scroll or {"y": 0, "height": 2000},
             "actions": [dict(a) for a in actions],
             "marker": marker if marker is not None else [0.0, url, 0, 0, 1120, 780,
                                                          title, text],
             "page_key": page_key if page_key is not None else [0.0, url, 0, 0, 1120, 780, []],
             "guards": guards or {}, "omitted_actions": 0}
    state["fingerprint"] = webagent.fingerprint(state)
    return state


# One text box (offered as a type *and* as a click to open it), one button, and the two
# appended controls. This is what a real search page hands the model.
CONTROLS = [action("e1", 1, "fill", "Query", role="textbox"),
            action("e2", 1, "click", "Open Query", role="textbox"),
            action("e3", 3, "click", "Search"),
            appended("scroll_down", "scroll", "Scroll down", delta=560),
            appended("wait", "wait", "Wait for the page to update")]
SEARCH = page(actions=CONTROLS)
TYPED = page(actions=[dict(a, value="titanium drill") if a.get("node") == 1 else dict(a)
                      for a in CONTROLS])
RESULTS = page(url="https://shop.test/search?q=titanium+drill",
               text="1 result: titanium drill",
               actions=[action("e1", 4, "click", "titanium drill", role="link"),
                        appended("wait", "wait", "Wait for the page to update")])


def scripted(script, *, goal_probability=0.9, cost_usd=0.0004, tokens=2_000):
    """A stand-in decider with the same return shape as :func:`webagent.decide`.

    Entries are offered action ids, or DONE/BLOCKED. The last entry repeats, so
    ``["e1", "DONE"]`` describes a one-action task that then claims to be finished.
    """
    calls = []

    def decider(page_state, goal, history, *, model="", key="", endpoint=""):
        chosen = script[min(len(calls), len(script) - 1)]
        calls.append({"page": page_state, "goal": goal, "steps": len(history),
                      "model": model, "key": key, "endpoint": endpoint})
        offered = next((a for a in page_state["actions"] if a["id"] == chosen), None)
        if offered is None and chosen not in {"DONE", "BLOCKED"}:
            raise AssertionError(f"scripted choice {chosen!r} is not offered by this page")
        return {"choice": chosen,
                "operation": (webagent.KIND_TO_OPERATION.get(offered["kind"], "OTHER")
                              if offered else chosen),
                "target": "1", "confidence": 0.91, "probabilities": {chosen: 0.91},
                "operation_probabilities": {chosen: 0.91}, "target_probabilities": {},
                "target_confidence": None, "goal_satisfied": goal_probability,
                "raw_answers": {}, "model": model or "d1.2",
                "usage": {"input_tokens": tokens, "output_tokens": 0},
                "cost_usd": cost_usd, "latency_ms": 240, "request_bytes": 5_000,
                "fingerprint": page_state["fingerprint"]}

    decider.calls = calls
    return decider


class FakeBrowser:
    """The four methods ``WebTask`` uses, over a list of page states."""

    def __init__(self, pages):
        self.pages = list(pages)
        self.index = 0
        self.acted = []
        self.observed = 0
        self.closed = False
        self.refuse_in_act = 0
        self.fail_observe = 0

    @property
    def current(self):
        return self.pages[min(self.index, len(self.pages) - 1)]

    def observe(self, screenshot=False):
        if self.fail_observe:
            self.fail_observe -= 1
            raise webagent.StalePage("Document is navigating")
        self.observed += 1
        state = dict(self.current)
        state["actions"] = [dict(a) for a in self.current["actions"]]
        if screenshot:
            state["screenshot"] = "jpeg"
        return state

    def fresh(self, page_state, action=None):
        return page_state["fingerprint"] == self.current["fingerprint"]

    def act(self, action, page_state, text=None):
        if self.refuse_in_act:
            self.refuse_in_act -= 1
            raise webagent.StalePage("Page changed since this decision. Observe again.")
        self.acted.append((action["id"], action["kind"], text))
        if action["kind"] != "wait":
            self.index += 1
        return {"executed": action["id"]}

    def close(self):
        self.closed = True


def make_task(*, pages=None, script=None, goal="find a titanium drill", writer=None,
              goal_probability=0.9, cost_usd=0.0004, tokens=2_000, **kwargs):
    browser = FakeBrowser(pages if pages is not None else [SEARCH, TYPED, RESULTS])
    decider = scripted(script if script is not None else ["e1", "e3", "DONE"],
                       goal_probability=goal_probability, cost_usd=cost_usd, tokens=tokens)
    environ = {"LIQUID_API_KEY": "liquid_test_key", "LIQUID_MODEL": "d1.2"}
    task = webagent.WebTask("https://shop.test/search", goal, browser=browser, decider=decider,
                            writer=writer or (lambda context: ("titanium drill", {"usage": {}})),
                            environ=environ, **kwargs)
    return task, browser, decider


class Offerings(unittest.TestCase):
    """What a page becomes: one index per element, one target set per operation."""

    def test_one_index_per_element_even_when_two_operations_share_it(self):
        elements, targets, controls = webagent.action_space(SEARCH["actions"])
        self.assertEqual([e["index"] for e in elements], ["1", "2"])
        self.assertEqual(elements[0]["operations"], ["TYPE_TEXT", "CLICK"])
        self.assertEqual(sorted(controls), ["SCROLL_DOWN", "WAIT"])
        self.assertEqual(sorted(targets["CLICK"]), ["1", "2"])
        self.assertEqual(sorted(targets["TYPE_TEXT"]), ["1"])
        self.assertNotIn("SELECT", targets)

    def test_a_dropdown_offers_its_options_not_the_control(self):
        actions = [action("e1", 7, "select", "Size → Medium", role="combobox",
                          value="medium", current_value="Small"),
                   action("e2", 7, "select", "Size → Large", role="combobox",
                          value="large", current_value="Small")]
        elements, targets, _controls = webagent.action_space(actions)
        self.assertEqual(elements[0]["options"], [
            {"index": "1:1", "label": "Size → Medium", "value": "medium"},
            {"index": "1:2", "label": "Size → Large", "value": "large"}])
        self.assertEqual(elements[0]["value"], "Small", "the model must see what is chosen now")
        self.assertEqual(sorted(targets["SELECT"]), ["1:1", "1:2"])

    def test_questions_are_an_operation_head_a_goal_check_and_one_target_head_per_operation(self):
        elements, targets, controls = webagent.action_space(SEARCH["actions"])
        questions, _operations = webagent.build_questions("find a titanium drill", elements,
                                                          targets, controls)
        self.assertEqual(sorted(questions), ["click_target", "goal_satisfied", "operation",
                                             "type_text_target"])
        self.assertEqual(questions["goal_satisfied"]["type"], "noul")
        self.assertEqual(sorted(questions["operation"]["criteria"]),
                         ["BLOCKED", "CLICK", "DONE", "SCROLL_DOWN", "TYPE_TEXT", "WAIT"])
        for name in ("operation", "click_target", "type_text_target"):
            self.assertIn("find a titanium drill", questions[name]["instructions"])
        # The offer text is all d1.2 reads, so the element's own words have to be in it.
        self.assertEqual(questions["click_target"]["criteria"]["2"]["element"], "[2] Search")

    def test_the_question_set_fits_one_documented_call(self):
        # One billable round trip per step is the design; check_questions is what refuses
        # a page that would need two, on the way to the documented 128-question ceiling.
        elements, targets, controls = webagent.action_space(SEARCH["actions"])
        questions, _ = webagent.build_questions("goal", elements, targets, controls)
        self.assertLessEqual(len(liquid.check_questions(questions)), liquid.MAX_QUESTIONS)

    def test_state_carries_the_page_the_elements_and_the_recent_history(self):
        history = [{"action": "a", "kind": "click", "text": None, "page_changed": True}] * 20
        state = webagent.build_state(SEARCH, "goal", [{"index": "1"}], history)
        self.assertEqual(sorted(state), ["elements", "page", "recent_actions"])
        self.assertEqual(sorted(state["page"]), ["text", "title", "url"])
        self.assertEqual(len(state["recent_actions"]), webagent.HISTORY_CONTEXT)


class Geometry(unittest.TestCase):
    def test_a_relayout_is_not_a_page_change(self):
        moved = page(actions=[action("e1", 1, "click", "Search",
                                     rect={"x": 0, "y": 900, "w": 50, "h": 50})])
        still = page(actions=[action("e1", 1, "click", "Search",
                                     rect={"x": 12, "y": 34, "w": 88, "h": 16})])
        self.assertEqual(moved["fingerprint"], still["fingerprint"])

    def test_a_new_value_a_new_page_and_a_scroll_are_page_changes(self):
        self.assertNotEqual(SEARCH["fingerprint"], TYPED["fingerprint"])
        self.assertNotEqual(TYPED["fingerprint"], RESULTS["fingerprint"])
        scrolled = page(actions=CONTROLS, scroll={"y": 560, "height": 2000})
        self.assertNotEqual(SEARCH["fingerprint"], scrolled["fingerprint"])


class Decision(unittest.TestCase):
    """The real client, with the real questions, over a scripted transport."""

    class FakeResponse:
        def __init__(self, payload):
            self._raw = json.dumps(payload).encode()

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return self._raw

    class FakeOpener:
        def __init__(self, payload):
            self.payload = payload
            self.calls = []

        def open(self, request, timeout=None):
            self.calls.append({"url": request.full_url, "body": json.loads(request.data)})
            return Decision.FakeResponse(self.payload)

    OPERATIONS = ["CLICK", "TYPE_TEXT", "SCROLL_DOWN", "WAIT", "DONE", "BLOCKED"]

    def choice(self, selected, probabilities):
        return {"type": "choice", "choice": selected, "probabilities": probabilities,
                "confidence": max(probabilities.values())}

    def operation_answer(self, winner):
        others = round(0.3 / (len(self.OPERATIONS) - 1), 6)
        return self.choice(winner, {key: (0.7 if key == winner else others)
                                    for key in self.OPERATIONS})

    def answers(self, operation="CLICK", *, click="2", goal=0.1, type_text=None,
                click_head=None):
        return {
            "operation": self.operation_answer(operation),
            "click_target": click_head or self.choice(click, {"1": 0.1, "2": 0.9}),
            "type_text_target": type_text or self.choice("1", {"1": 1.0}),
            "goal_satisfied": {"type": "noul", "noul": goal},
        }

    def decide(self, answers, **kwargs):
        opener = self.FakeOpener({"model": "d1.2",
                                  "usage": {"input_tokens": 3_000, "output_tokens": 0},
                                  "answers": answers})
        result = webagent.decide(SEARCH, "find a titanium drill", [], key="liquid_test_key",
                                 environ={"LIQUID_MODEL": "d1.2"}, opener=opener, **kwargs)
        return result, opener

    def test_the_winners_are_resolved_back_to_observed_ids(self):
        result, opener = self.decide(self.answers())
        self.assertEqual(opener.calls[0]["url"], liquid.ENDPOINT)
        body = opener.calls[0]["body"]
        self.assertEqual(sorted(body), ["model", "questions", "state"])
        self.assertEqual(body["model"], "d1.2")
        self.assertEqual(body["state"]["page"]["url"], "https://shop.test/search")
        self.assertEqual(result["choice"], "e3")
        self.assertEqual(result["operation"], "CLICK")
        self.assertEqual(result["target"], "2")
        self.assertEqual(result["probabilities"], {"e2": 0.1, "e3": 0.9})
        self.assertEqual(result["goal_satisfied"], 0.1)
        self.assertEqual(result["usage"], {"input_tokens": 3_000, "output_tokens": 0})
        self.assertEqual(result["cost_usd"], 0.00006)
        self.assertEqual(result["fingerprint"], SEARCH["fingerprint"])

    def test_an_unselected_operation_head_cannot_block_the_step(self):
        # Only the winning operation's target head can cause an action, so a nonsense
        # answer in the losing head is ignored rather than fatal or retried.
        broken = self.choice("9", {"1": 0.4, "9": 0.6})
        result, _opener = self.decide(self.answers(type_text=broken))
        self.assertEqual(result["choice"], "e3")

    def test_an_unoffered_index_in_the_winning_head_refuses_the_step(self):
        with self.assertRaises(liquid.LiquidError):
            self.decide(self.answers(click="7"))

    def test_done_is_an_operation_not_a_target(self):
        result, _opener = self.decide(self.answers("DONE", goal=0.97))
        self.assertEqual(result["choice"], "DONE")
        self.assertIsNone(result["target"])
        self.assertEqual(result["probabilities"], {"DONE": 0.7})

    def test_a_control_choice_keeps_its_observed_id(self):
        result, _opener = self.decide(self.answers("SCROLL_DOWN"))
        self.assertEqual(result["choice"], "scroll_down")

    def test_an_unusable_goal_answer_is_refused_before_anything_acts(self):
        with self.assertRaises(liquid.LiquidError):
            self.decide(self.answers(goal=1.7))

    def test_a_missing_head_answer_is_refused_not_defaulted(self):
        answers = self.answers()
        del answers["goal_satisfied"]
        with self.assertRaises(liquid.LiquidError):
            self.decide(answers)


class Loop(unittest.TestCase):
    def run_task(self, **kwargs):
        task, browser, decider = make_task(**kwargs)
        return list(task.run()), task, browser, decider

    def test_it_fills_from_the_writer_clicks_and_then_finishes(self):
        snapshots, _task, browser, decider = self.run_task()
        self.assertEqual([s["status"] for s in snapshots], ["ready", "ready", "done"])
        self.assertEqual(browser.acted, [("e1", "fill", "titanium drill"),
                                         ("e3", "click", None)])
        history = snapshots[-1]["history"]
        self.assertEqual([h["kind"] for h in history], ["fill", "click"])
        self.assertEqual([h["page_changed"] for h in history], [True, True])
        self.assertEqual([h["step"] for h in history], [1, 2])
        self.assertEqual(history[0]["text"], "titanium drill")
        self.assertEqual(snapshots[-1]["reason"], "")
        self.assertEqual(decider.calls[0]["key"], "liquid_test_key")
        self.assertEqual(decider.calls[1]["steps"], 1)
        # The second decision sees the value the first action actually put in the field.
        self.assertEqual(decider.calls[1]["page"]["actions"][0]["value"], "titanium drill")

    def test_every_call_is_billed_and_reported(self):
        _snapshots, task, _browser, _decider = self.run_task()
        self.assertEqual(task.calls, 3)
        self.assertAlmostEqual(task.spent_usd, 0.0012, places=8)
        self.assertEqual(task.tokens, 6_000)
        report = task.report()
        self.assertEqual(report["calls"], 3)
        self.assertEqual(report["cost_usd"], 0.0012, msg="the report rounds")
        self.assertEqual(report["input_tokens"], 6_000)

    def test_the_ceiling_stops_the_task_before_the_action_it_paid_for_runs(self):
        _snapshots, task, browser, _decider = self.run_task(max_cost_usd=0.001)
        self.assertEqual(task.status, "blocked")
        self.assertIn("$0.0010 ceiling", task.reason)
        self.assertEqual(task.calls, 3)
        self.assertEqual(len(browser.acted), 2)

    def test_a_refused_done_cannot_loop_forever(self):
        # DONE below the confirmation threshold adds nothing to the history, so the
        # decision counter - not the action counter - is what bounds it.
        _snapshots, task, browser, _decider = self.run_task(script=["DONE"], max_steps=3,
                                                            goal_probability=0.2)
        self.assertEqual(task.status, "blocked")
        self.assertEqual(task.reason, "reached the 3-decision budget")
        self.assertEqual(task.calls, 3)
        self.assertEqual(browser.acted, [])

    def test_the_action_budget_bounds_a_long_task(self):
        # Every step has to change the page, or the unchanged-page rule stops the task
        # first and the action budget is never the thing that bounds it.
        steps = [page(actions=CONTROLS, text=f"result {count}") for count in range(1, 7)]
        _snapshots, task, browser, _decider = self.run_task(pages=steps, script=["e3"],
                                                            max_steps=5)
        self.assertEqual(task.status, "blocked")
        self.assertEqual(task.reason, "reached the 5-action budget")
        self.assertEqual(len(browser.acted), 5)

    def test_a_model_that_repeats_one_control_is_stopped_even_when_the_page_churns(self):
        # Measured on the free open-weight d1-2B: it typed the query, selected the size,
        # then clicked Search 37 more times. Each click appended a line to the page's own
        # event log, so the fingerprint changed every step and the unchanged-page rule
        # never fired; only the 40-action budget did. .local-docs/web-probe/EVIDENCE.md
        # has the trace, and the reason the guard counts repetitions instead of narrowing
        # the offer is the ablation in withdraw.log.
        steps = [page(actions=CONTROLS, text=f"result {count}") for count in range(1, 12)]
        _snapshots, task, browser, _decider = self.run_task(pages=steps, script=["e3"])
        self.assertEqual(task.status, "blocked")
        self.assertEqual(task.reason, "chose 'Search' 8 times in a row")
        self.assertEqual(len(browser.acted), webagent.REPEAT_LIMIT)

    def test_repeating_a_scroll_is_not_mistaken_for_a_loop(self):
        # Reading a long page is legitimately repetitive: the same control, over and over,
        # is the whole task. Only the three input operations can trip the repeat rule.
        steps = [page(actions=CONTROLS, text=f"section {count}") for count in range(1, 12)]
        budget = webagent.REPEAT_LIMIT + 2
        _snapshots, task, browser, _decider = self.run_task(
            pages=steps, script=["scroll_down"], max_steps=budget)
        self.assertEqual(task.reason, f"reached the {budget}-action budget")
        self.assertEqual(len(browser.acted), budget)

    def test_done_needs_no_action_of_its_own(self):
        _snapshots, task, browser, _decider = self.run_task(script=["DONE"],
                                                            goal_probability=0.98)
        self.assertEqual(task.status, "done")
        self.assertEqual(task.reason, "")
        self.assertEqual(browser.acted, [])

    def test_a_low_goal_probability_refuses_done_without_ending_the_task(self):
        task, _browser, _decider = make_task(script=["e1", "DONE"], goal_probability=0.2)
        self.assertEqual(task.tick()["status"], "ready")
        refused = task.tick()
        self.assertEqual(refused["status"], "ready")
        self.assertEqual(refused["reason"], "DONE refused at goal probability 0.200")
        self.assertEqual(refused["steps"], 1)

    def test_a_model_that_acts_without_changing_the_page_is_stopped(self):
        same = page(actions=CONTROLS)
        _snapshots, task, browser, _decider = self.run_task(pages=[same], script=["e3"])
        self.assertEqual(task.status, "blocked")
        self.assertEqual(task.reason, "three actions in a row left the page unchanged")
        self.assertEqual(len(browser.acted), 3)
        self.assertEqual([h["page_changed"] for h in task.history], [False] * 3)

    def test_a_blocked_decision_ends_the_task(self):
        _snapshots, task, browser, _decider = self.run_task(script=["BLOCKED"])
        self.assertEqual(task.status, "blocked")
        self.assertEqual(task.reason, "no supported operation can progress")
        self.assertEqual(browser.acted, [])

    def test_a_moved_target_is_re_observed_and_never_double_applied(self):
        task, browser, decider = make_task(script=["e3"])
        task.predict()
        browser.refuse_in_act = 1
        snapshot = task.tick()
        self.assertEqual(browser.acted, [])
        self.assertEqual(snapshot["decision"], None)
        self.assertEqual(snapshot["status"], "ready")
        self.assertEqual(decider.calls[0]["steps"], 0)

    def test_a_decision_about_another_page_is_not_executed(self):
        task, browser, _decider = make_task(script=["e1"])
        decision = dict(task.predict())
        task.observe()
        decision["fingerprint"] = "a-sha-of-a-page-no-longer-showing"
        with self.assertRaises(webagent.StalePage):
            task.act(decision)
        self.assertEqual(browser.acted, [])

    def test_nothing_is_typed_when_the_local_engine_gives_no_value(self):
        def unreachable(context):
            raise webagent.NoFieldValue("the local engine gave no field value; "
                                        "nothing was typed")

        _snapshots, task, browser, _decider = self.run_task(script=["e1"], writer=unreachable)
        self.assertEqual(task.status, "blocked")
        self.assertEqual(task.reason, "the local engine gave no field value; nothing was typed")
        self.assertEqual(browser.acted, [])
        self.assertEqual(task.history, [])

    def test_a_refused_answer_ends_the_task_with_the_reason_not_a_traceback(self):
        def refused(page_state, goal, history, *, model="", key="", endpoint=""):
            raise liquid.LiquidError("d1.2 returned HTTP 401 authentication_error; "
                                     "no decision was made")

        browser = FakeBrowser([SEARCH])
        task = webagent.WebTask("https://shop.test/search", "goal", browser=browser,
                                decider=refused, environ={"LIQUID_API_KEY": "k"})
        list(task.run())
        self.assertEqual(task.status, "blocked")
        self.assertIn("HTTP 401", task.reason)
        self.assertEqual(browser.acted, [])

    def test_an_aborted_step_reuses_the_value_it_already_paid_for(self):
        asked = []

        def writer(context):
            asked.append((context["field"]["label"], context["field"]["value"]))
            return "titanium drill", {"usage": {}}

        same = page(actions=CONTROLS)
        task, browser, _decider = make_task(pages=[same, same, RESULTS],
                                            script=["e1", "e1", "DONE"], writer=writer)
        browser.refuse_in_act = 1
        self.assertEqual(task.tick()["status"], "ready", "the first attempt was aborted")
        self.assertEqual(asked, [("Query", "")])
        self.assertEqual(browser.acted, [])
        self.assertEqual(task.tick()["status"], "ready", "the retry went through")
        self.assertEqual(asked, [("Query", "")], "the local engine is not asked twice")
        self.assertEqual(browser.acted, [("e1", "fill", "titanium drill")])
        self.assertIsNone(task.pending_text, "a typed value is not offered to the next field")

    def test_a_terminated_task_refuses_another_prediction(self):
        task, _browser, _decider = make_task(script=["DONE"], goal_probability=0.99)
        list(task.run())
        with self.assertRaises(ValueError):
            task.predict()

    def test_the_report_keeps_the_outlet_contract_and_adds_the_bill(self):
        task, _browser, _decider = make_task()
        list(task.run())
        report = task.report()
        # The keys the outlet and both shells read out of every browser report.
        for key in ("ok", "steps", "status", "elapsed_ms", "url", "title", "decision",
                    "history", "text"):
            self.assertIn(key, report)
        self.assertTrue(report["ok"])
        self.assertEqual(report["engine"], "liquid-d1.2")
        self.assertEqual(report["model"], "d1.2")
        self.assertEqual(report["status"], "done")
        self.assertEqual(report["steps"], 2)
        self.assertEqual(report["url"], RESULTS["url"])
        self.assertEqual(report["title"], RESULTS["title"])
        self.assertEqual(report["decision"], None)
        self.assertEqual(report["max_cost_usd"], webagent.DEFAULT_MAX_COST_USD)
        self.assertLessEqual(len(report["history"]), webagent.REPORT_HISTORY)
        self.assertIn("titanium drill", report["text"])

    def test_a_run_that_never_reached_a_terminal_state_is_not_ok(self):
        task, _browser, _decider = make_task(script=["e1"])
        task.tick()
        self.assertFalse(task.report()["ok"])
        self.assertEqual(task.report()["status"], "ready")

    def test_the_credentials_come_from_the_environment_before_a_browser_is_touched(self):
        browser = FakeBrowser([SEARCH])
        with self.assertRaises(liquid.LiquidError) as caught:
            webagent.WebTask("https://shop.test/search", "goal", browser=browser,
                             decider=scripted(["DONE"]), environ={})
        self.assertIn("LIQUID_API_KEY", str(caught.exception))
        self.assertEqual(browser.observed, 0)

    def test_the_ceiling_honours_an_environment_override(self):
        browser = FakeBrowser([SEARCH])
        task = webagent.WebTask("https://shop.test/search", "goal", browser=browser,
                                decider=scripted(["DONE"]),
                                environ={"LIQUID_API_KEY": "k",
                                         "FENNEC_WEB_MAX_COST": "0.5"})
        self.assertEqual(task.ceiling, 0.5)

    def test_screenshots_are_taken_only_when_asked_for(self):
        plain = webagent.WebTask("https://shop.test/search", "goal", browser=FakeBrowser([SEARCH]),
                                 decider=scripted(["DONE"]), environ={"LIQUID_API_KEY": "k"})
        self.assertNotIn("screenshot", plain.page)
        shot = webagent.WebTask("https://shop.test/search", "goal",
                                browser=FakeBrowser([SEARCH]), decider=scripted(["DONE"]),
                                environ={"LIQUID_API_KEY": "k"}, screenshots=True)
        self.assertEqual(shot.page["screenshot"], "jpeg")

    def test_the_goal_is_the_only_mandatory_argument(self):
        with self.assertRaises(ValueError):
            webagent.WebTask("https://shop.test/search", "   ", browser=FakeBrowser([SEARCH]))

    def test_closing_the_task_closes_the_tab(self):
        browser = FakeBrowser([SEARCH])
        with webagent.WebTask("https://shop.test/search", "goal", browser=browser,
                              decider=scripted(["DONE"], goal_probability=0.9),
                              environ={"LIQUID_API_KEY": "k"}):
            pass
        self.assertTrue(browser.closed)

    def test_a_page_that_never_arrived_still_closes_the_tab(self):
        # The browser is opened inside the constructor, so a caller that only got an
        # exception back has no task to close. One stranded tab per failed attempt is
        # how a retrying caller loses a browser it never asked to keep.
        browser = FakeBrowser([SEARCH])
        browser.fail_observe = 1
        with self.assertRaises(webagent.StalePage):
            webagent.WebTask("https://shop.test/search", "goal", browser=browser,
                             environ={"LIQUID_API_KEY": "k"})
        self.assertTrue(browser.closed)


THREW = object()
GUARD = ["1", "button", "Search", None, None, None, None, False, None, None, None, None, None,
         "results"]


class FakeCDP:
    """Answers the CDP methods the transport uses and records every call.

    ``Runtime.evaluate`` is matched by substring, so a test says what the page's
    JavaScript answered without reimplementing the page. The reader and the marker probe
    share text (the marker wraps the reader), so needles go from most to least specific.
    """

    DEFAULTS = {"Target.createTarget": {"targetId": "T"},
                "Target.attachToTarget": {"sessionId": "S"},
                "Page.captureScreenshot": {"data": "jpeg"},
                "Target.closeTarget": {}}
    # Opening a browser polls until the document reports complete. Left unanswered that
    # poll runs for the whole navigation timeout, so every test that opens one would pay
    # it in wall clock; answering it here means a test only overrides it when it wants to.
    NAVIGATED = ("document.readyState", "complete")

    def __init__(self, evaluated=None, **responses):
        self.evaluated = list(evaluated or []) + [self.NAVIGATED]
        self.responses = responses
        self.calls = []

    def __call__(self, method, session_id=None, **params):
        self.calls.append({"method": method, "session_id": session_id, **params})
        if method == "Runtime.evaluate":
            expression = params["expression"]
            for needle, value in self.evaluated:
                if needle in expression:
                    value = value(params) if callable(value) else value
                    if value is THREW:
                        return {"exceptionDetails": {"text": "SyntaxError"}}
                    return {"result": {"value": value}}
            return {"result": {"value": None}}
        value = self.responses.get(method, self.DEFAULTS.get(method, {}))
        return value(params) if callable(value) else value

    def where(self, method):
        return [call for call in self.calls if call["method"] == method]

    def input_events(self):
        return [c["type"] for c in self.where("Input.dispatchMouseEvent")]


class Transport(unittest.TestCase):
    def open(self, cdp, url="https://shop.test/", **kwargs):
        started = []
        browser = webagent.Browser(url, cdp=cdp, ensure_daemon=lambda: started.append(1),
                                   sleep=lambda _seconds: None, **kwargs)
        return browser, started

    def clickable(self, *, kind="click", label="Search", node=1, **extra):
        """A one-action page whose guard the transport can re-read as unchanged."""
        state = page(actions=[action("e1", node, kind, label, **extra)],
                     page_key=["key"], guards={"1": GUARD})
        return state, state["actions"][0]

    def test_it_opens_one_background_tab_at_a_fixed_viewport_and_navigates_once(self):
        cdp = FakeCDP([("readyState", "complete")],
                      **{"Target.createTarget": {"targetId": "T"},
                         "Target.attachToTarget": {"sessionId": "S"}})
        browser, started = self.open(cdp)
        self.assertEqual([c["method"] for c in cdp.calls],
                         ["Target.createTarget", "Target.attachToTarget",
                          "Emulation.setDeviceMetricsOverride",
                          "Emulation.setFocusEmulationEnabled", "Page.navigate",
                          "Runtime.evaluate"])
        self.assertEqual(started, [1], "the daemon is what keeps one browser for the task")
        self.assertEqual((browser.target, browser.session), ("T", "S"))
        create = cdp.calls[0]
        self.assertEqual((create["url"], create["background"]), ("about:blank", True))
        self.assertIsNone(create["session_id"], "the tab is created on the browser session")
        metrics = cdp.calls[2]
        self.assertEqual((metrics["width"], metrics["height"], metrics["mobile"]),
                         (1120, 780, False))
        self.assertTrue(cdp.calls[4]["url"].startswith("https://"))
        self.assertEqual({c["session_id"] for c in cdp.calls[2:]}, {"S"})

    def test_it_only_browses_http(self):
        cdp = FakeCDP()
        with self.assertRaises(ValueError):
            self.open(cdp, url="file:///C:/Users/me/inbox/index.html")
        self.assertEqual(cdp.calls, [])

    def test_a_missing_transport_names_the_package(self):
        saved = {name: module for name, module in sys.modules.items()
                 if name.split(".")[0] == "browser_harness"}
        for name in saved:
            del sys.modules[name]
        sys.modules["browser_harness"] = None
        try:
            with self.assertRaises(webagent.BrowserUnavailable) as caught:
                webagent._transport()
        finally:
            del sys.modules["browser_harness"]
            sys.modules.update(saved)
        self.assertIn("browser-harness", str(caught.exception))

    def test_an_expression_that_throws_is_a_stale_page_not_a_crash(self):
        browser, _started = self.open(FakeCDP([("throwing", THREW)]))
        with self.assertRaises(webagent.StalePage):
            browser.evaluate("throwing")

    def test_observation_fingerprints_the_page_and_the_screenshot_is_optional(self):
        state = page(actions=[])
        expected = webagent.fingerprint(state)
        browser, _started = self.open(FakeCDP([("const cache", state)],
                                              **{"Page.captureScreenshot": {"data": "jpeg"}}))
        self.assertNotIn("screenshot", browser.observe(screenshot=False))
        read = browser.observe(screenshot=True)
        self.assertEqual(read["screenshot"], "jpeg")
        self.assertEqual(read["fingerprint"], expected)

    def test_an_empty_document_is_a_navigation_not_an_empty_page(self):
        browser, _started = self.open(FakeCDP([("const cache", None)]))
        with self.assertRaises(webagent.StalePage):
            browser.observe()

    def test_freshness_of_a_targeted_action_re_reads_that_node_alone(self):
        state, first = self.clickable()
        asked = []

        def answer(params):
            asked.append(params["expression"])
            return [["key"], GUARD]

        browser, _started = self.open(FakeCDP([("c.nodes.get", answer)]))
        self.assertTrue(browser.fresh(state, first))
        self.assertEqual(len(asked), 1)
        self.assertIn("c.nodes.get(1)", asked[0])
        self.assertNotIn("nodes.get(2)", asked[0])

        browser, _started = self.open(FakeCDP([("c.nodes.get", [["key"], ["2", "button"]])]))
        self.assertFalse(browser.fresh(state, first), "the label or state changed under us")

    def test_freshness_without_a_target_compares_the_whole_page_marker(self):
        state = page(actions=CONTROLS)
        browser, _started = self.open(FakeCDP([("state?.marker", state["marker"])]))
        self.assertTrue(browser.fresh(state))
        browser, _started = self.open(FakeCDP([("state?.marker", ["other", "origin"])]))
        self.assertFalse(browser.fresh(state))

    def test_a_hit_tested_click_is_exactly_two_mouse_events_at_the_resolved_point(self):
        state, first = self.clickable()
        browser, _started = self.open(FakeCDP([("c.nodes.get", [["key"], GUARD]),
                                               ("elementFromPoint", {"x": 60, "y": 20})]))
        self.assertEqual(browser.act(first, state), {"executed": "e1"})
        events = browser.cdp.where("Input.dispatchMouseEvent")
        self.assertEqual([e["type"] for e in events], ["mousePressed", "mouseReleased"])
        self.assertEqual({(e["x"], e["y"]) for e in events}, {(60, 20)})
        self.assertEqual({e["button"] for e in events}, {"left"})
        self.assertEqual({e["clickCount"] for e in events}, {1})

    def test_the_page_is_handed_the_observed_id_and_nothing_else(self):
        state, first = self.clickable()
        browser, _started = self.open(FakeCDP([("c.nodes.get", [["key"], GUARD]),
                                               ("elementFromPoint", {"x": 1, "y": 1})]))
        browser.act(first, state)
        evaluated = [c["expression"] for c in browser.cdp.where("Runtime.evaluate")]
        resolution = [e for e in evaluated if "elementFromPoint" in e][0]
        self.assertIn(json.dumps(first), resolution)
        # The page's only lever is the id it was read from. No selector, coordinate, URL
        # or line of code crosses this boundary, so nothing a page says can be executed.
        for forbidden in ("querySelector", "css:", "document.write", "eval("):
            self.assertNotIn(forbidden, resolution)

    def test_a_covered_or_detached_target_is_never_clicked(self):
        state, first = self.clickable()
        browser, _started = self.open(FakeCDP([("c.nodes.get", [["key"], GUARD]),
                                               ("elementFromPoint", None)]))
        with self.assertRaises(webagent.StalePage):
            browser.act(first, state)
        self.assertEqual(browser.cdp.input_events(), [])
        self.assertIsNone(browser.after_input)

    def test_a_select_is_applied_by_value_and_never_retried_on_doubt(self):
        state, first = self.clickable(kind="select", label="Size → Large", role="combobox",
                                      node=7, value="large")
        state["guards"]["7"] = ["7", "combobox", "Size", "large", None, 1, None, False,
                                None, None, None, None, None, "Size"]
        browser, _started = self.open(FakeCDP([("c.nodes.get", [["key"], state["guards"]["7"]]),
                                               ("elementFromPoint", {"x": 5, "y": 5})]))
        browser.act(first, state)
        self.assertEqual(browser.cdp.input_events(), [], "a select needs no mouse click")

        refused, _ = self.open(FakeCDP([("c.nodes.get", [["key"], state["guards"]["7"]]),
                                        ("elementFromPoint", None)]))
        with self.assertRaises(RuntimeError) as caught:
            refused.act(first, state)
        self.assertIn("not confirmed", str(caught.exception))

        interrupted, _ = self.open(FakeCDP([("c.nodes.get", [["key"], state["guards"]["7"]]),
                                            ("elementFromPoint", THREW)]))
        with self.assertRaises(RuntimeError) as caught:
            interrupted.act(first, state)
        self.assertIn("interrupted", str(caught.exception))

    def test_a_fill_focuses_selects_and_then_inserts(self):
        state, first = self.clickable(kind="fill", label="Query", role="textbox")
        # A field can be replaced wholesale between reads, so typing is cleared by the
        # whole-page marker rather than by one node's guard.
        browser, _started = self.open(FakeCDP([("state?.marker", state["marker"]),
                                               ("elementFromPoint", {"x": 3, "y": 4})]))
        browser.act(first, state, text="titanium drill")
        # The click is what gives the field focus; insertText only reaches the caret that
        # focus put there. Then the old contents are replaced, not appended to.
        self.assertEqual(browser.cdp.input_events(), ["mousePressed", "mouseReleased"])
        keys = browser.cdp.where("Input.dispatchKeyEvent")
        self.assertEqual([k["type"] for k in keys], ["keyDown", "keyUp"])
        self.assertEqual(keys[0]["commands"], ["selectAll"])
        self.assertEqual(browser.cdp.where("Input.insertText")[0]["text"], "titanium drill")
        self.assertEqual(browser.after_input["id"], "e1")

    def test_a_wait_touches_no_input_and_leaves_the_page_alone(self):
        slept = []
        state = page(actions=[appended("wait", "wait", "Wait for the page to update")])
        browser, _started = self.open(FakeCDP([("state?.marker", state["marker"])]))
        browser._sleep = slept.append
        self.assertEqual(browser.act(state["actions"][0], state), {"executed": "wait"})
        self.assertEqual(slept, [0.1])
        self.assertEqual(browser.cdp.input_events(), [])
        self.assertIsNone(browser.after_input)

    def test_a_scroll_wheels_the_centre_of_the_viewport(self):
        state = page(actions=[appended("scroll_down", "scroll", "Scroll down", delta=560)])
        browser, _started = self.open(FakeCDP([("state?.marker", state["marker"])]))
        browser.act(state["actions"][0], state)
        wheel = browser.cdp.where("Input.dispatchMouseEvent")[0]
        self.assertEqual((wheel["type"], wheel["deltaY"], wheel["deltaX"]),
                         ("mouseWheel", 560, 0))
        self.assertEqual((wheel["x"], wheel["y"]), (560, 650))

    def test_a_fabricated_node_reference_is_refused(self):
        forged = action("e1", "#evil", "click", "Search")
        browser, _started = self.open(FakeCDP([("c.nodes.get", [["key"], GUARD])]))
        with self.assertRaises(ValueError):
            browser.act(forged, page(actions=[forged]))
        self.assertEqual(browser.cdp.input_events(), [])

    def test_closing_is_idempotent(self):
        browser, _started = self.open(FakeCDP())
        browser.close()
        browser.close()
        self.assertEqual(len(browser.cdp.where("Target.closeTarget")), 1)


class FieldValues(unittest.TestCase):
    """The one place text is written rather than chosen: the local engine decides it."""

    def ask(self, raw, context=None):
        import engine
        real = engine.chat
        seen = {}

        def fake(messages, **kwargs):
            seen.update(messages=messages, **kwargs)
            return {"role": "assistant", "content": raw}, {"input_tokens": 40}

        engine.chat = fake
        try:
            return webagent.default_writer(context or {"goal": "goal"}), seen
        finally:
            engine.chat = real

    def test_only_a_bare_text_object_is_accepted(self):
        for raw, expected in [('{"text": "ok"}', "ok"), ('{"text": "  spaced  "}', "  spaced  "),
                              ('{"text": "Ünïcode ✓"}', "Ünïcode ✓")]:
            (value, _meta), _seen = self.ask(raw)
            self.assertEqual(value, expected)
        for raw in ('{"text": null}', '{"text": ""}', '{"text": "   "}', '{"text": 7}',
                    '{"text": {"a": 1}}', '{"value": "x"}', '{"text": "x", "note": "y"}',
                    "not json", '{"text": "' + "x" * (webagent.MAX_FIELD_CHARS + 1) + '"}'):
            with self.subTest(raw=raw[:40]):
                import engine
                real = engine.chat

                def fake(messages, **kwargs):
                    return {"role": "assistant", "content": raw}, {}

                engine.chat = fake
                try:
                    with self.assertRaises(webagent.NoFieldValue):
                        webagent.default_writer({"goal": "goal"})
                finally:
                    engine.chat = real

    def test_an_unreachable_engine_types_nothing_and_says_why(self):
        import engine
        real = engine.chat

        def unreachable(messages, **kwargs):
            raise engine.EngineError("engine is not running")

        engine.chat = unreachable
        try:
            with self.assertRaises(webagent.NoFieldValue) as caught:
                webagent.default_writer({"goal": "goal"})
        finally:
            engine.chat = real
        self.assertIn("nothing was typed", str(caught.exception))
        self.assertIn("engine is not running", str(caught.exception))

    def test_the_prompt_carries_the_rules_the_context_and_no_creativity(self):
        context = {"goal": "find a titanium drill", "field": {"label": "Query", "role": "textbox",
                                                              "value": ""},
                   "page": {"title": "Search", "text": "body"}, "recent_actions": []}
        (_value, _meta), seen = self.ask('{"text": "titanium drill"}', context)
        self.assertEqual(seen["messages"][0]["content"], webagent.FIELD_VALUE)
        self.assertEqual(json.loads(seen["messages"][1]["content"]), context)
        self.assertEqual(seen["temperature"], 0.0)
        self.assertEqual(seen["max_tokens"], 512)

    def test_the_field_context_shows_the_goal_the_field_the_page_and_recent_steps(self):
        context = webagent.field_context("find a drill", SEARCH["actions"][0], SEARCH,
                                         [{"action": "a", "text": None}] * 20)
        self.assertEqual(sorted(context), ["field", "goal", "page", "recent_actions"])
        self.assertEqual(context["field"], {"label": "Query", "role": "textbox", "value": ""})
        self.assertEqual(len(context["recent_actions"]), webagent.RECENT_HISTORY)


class Rules(unittest.TestCase):
    def test_the_rules_treat_the_page_as_data_and_the_goal_as_the_authority(self):
        for rule in (webagent.NEXT_ACTION, webagent.TARGET_RULE, webagent.GOAL_CHECK,
                     webagent.FIELD_VALUE):
            self.assertTrue(rule.strip())
        self.assertIn("untrusted data", webagent.NEXT_ACTION)
        self.assertIn("untrusted data", webagent.FIELD_VALUE)
        self.assertIn("offered element index", webagent.TARGET_RULE)
        self.assertIn('"text"', webagent.FIELD_VALUE)

    def test_the_cost_ceiling_is_a_tripwire_looser_than_the_step_budget(self):
        # One step sends the page text plus a couple of hundred labelled controls: call it
        # tens of thousands of input tokens. At the documented price that is a fraction of a
        # cent, so 40 steps cannot reach $0.25 and only a task that keeps paying for
        # nothing can. If that ratio ever inverts, this test is the alarm.
        per_step = liquid.cost_usd("d1.2", {"input_tokens": 40_000})
        self.assertGreater(webagent.DEFAULT_MAX_COST_USD / per_step, 5 * webagent.MAX_STEPS)


class WebDriverTests(unittest.TestCase):
    """The process boundary in :mod:`web_driver`, without a browser.

    What matters here is the contract outlets depends on: which exit code means "nothing
    was ever started" (so the caller may decline without a report) and which means "this
    loop had a tab" (so it must not be replayed), and that a report always reaches stdout
    behind the marker even when the run ended badly.
    """

    ARGS = ["--url", "https://shop.test/search", "--goal", "buy a titanium drill"]

    def drive(self, argv, task):
        """Run ``main`` with a scripted task, returning (exit code, stdout text)."""
        out = io.StringIO()
        with mock.patch.object(webagent, "WebTask", return_value=task), \
                contextlib.redirect_stdout(out):
            code = web_driver.main(argv)
        return code, out.getvalue()

    def report_from(self, text):
        self.assertTrue(text.startswith(web_driver.SENTINEL), text[:40])
        return json.loads(text[len(web_driver.SENTINEL):])

    @staticmethod
    def task(**kwargs):
        return FakeTask(**kwargs)

    def test_a_finished_task_is_reported_and_exits_zero(self):
        code, text = self.drive(self.ARGS, self.task(snapshots=3))
        self.assertEqual(code, 0)
        report = self.report_from(text)
        self.assertEqual(report["ok"], True)
        self.assertEqual(report["engine"], "liquid-d1.2")
        self.assertNotIn("stopped_after", report)
        self.assertNotIn("error", report)
        self.assertNotIn("leak", report)

    def test_a_loop_that_cannot_be_imported_exits_2_without_a_tab(self):
        """The one exit code the outlet may decline on: nothing was ever started,
        so there is no report to return and no tab anybody left behind."""
        saved = sys.modules.get("webagent")
        sys.modules["webagent"] = None
        out = io.StringIO()
        try:
            with contextlib.redirect_stdout(out):
                code = web_driver.main(self.ARGS)
        finally:
            if saved is None:
                sys.modules.pop("webagent", None)
            else:
                sys.modules["webagent"] = saved
        self.assertEqual(code, 2)
        report = self.report_from(out.getvalue())
        self.assertEqual(report["ok"], False)
        self.assertIn("not importable", report["error"])

    def test_a_task_that_cannot_be_built_exits_3_because_a_tab_may_exist(self):
        """Constructing is what opens the tab, so this code must never fall back."""
        def refuse(*args, **kwargs):
            raise OSError("Chrome refused to start")

        out = io.StringIO()
        with mock.patch.object(webagent, "WebTask", refuse), contextlib.redirect_stdout(out):
            code = web_driver.main(self.ARGS)
        self.assertEqual(code, 3)
        report = self.report_from(out.getvalue())
        self.assertEqual(report, {"ok": False, "engine": "liquid-d1.2",
                                  "error": "OSError: Chrome refused to start"})

    def test_the_wall_clock_stops_the_loop_without_losing_the_accounting(self):
        task = self.task(snapshots=50, report={"ok": False, "engine": "liquid-d1.2",
                                               "status": "running", "steps": 2,
                                               "cost_usd": 0.01})
        # Two reads per step would drift past the deadline; this clock says the first step
        # was free and the second already blew the budget.
        clock = iter([0.0, 0.0, 30.0])
        with mock.patch.object(web_driver, "time", SimpleNamespace(monotonic=lambda: next(clock))):
            code, text = self.drive(self.ARGS + ["--timeout", "5"], task)
        report = self.report_from(text)
        self.assertEqual(report["stopped_after"], "the 5s wall clock")
        self.assertEqual(report["steps"], 2)
        self.assertEqual(code, 1)
        self.assertEqual(task.closed, 1)

    def test_a_loop_that_dies_mid_run_still_says_what_it_cost(self):
        task = self.task(snapshots=1, run_raises=RuntimeError("the page vanished"))
        code, text = self.drive(self.ARGS, task)
        report = self.report_from(text)
        self.assertEqual(report["ok"], False)
        self.assertEqual(report["error"], "RuntimeError: the page vanished")
        self.assertEqual(report["cost_usd"], task.report()["cost_usd"])
        self.assertEqual(code, 1)
        self.assertEqual(task.closed, 1, "a tab the loop left behind is still handed back")

    def test_a_tab_that_cannot_be_closed_is_named_not_discarded(self):
        task = self.task(snapshots=2, close_raises=OSError("the tab is already gone"))
        code, text = self.drive(self.ARGS, task)
        report = self.report_from(text)
        self.assertEqual(code, 0, "the paid-for answer stays the answer")
        self.assertEqual(report["ok"], True)
        self.assertEqual(report["leak"], "the tab could not be closed: OSError: the tab is already gone")

    def test_only_the_limits_asked_for_reach_the_loop(self):
        """Defaults belong to the loop; forwarding argparse's zeros would override them."""
        built = self.build(self.ARGS)
        self.assertEqual(built.kwargs, {"model": "", "screenshots": False, "progress": None})
        self.assertNotIn("max_steps", built.kwargs)
        self.assertNotIn("max_cost_usd", built.kwargs)

    def test_the_flags_reach_the_loop_as_the_loop_names_them(self):
        built = self.build(self.ARGS + ["--max-steps", "4", "--max-cost-usd", "0.05",
                                        "--model", "d1.2-ultra", "--screenshots"])
        self.assertEqual(built.args, ("https://shop.test/search", "buy a titanium drill"))
        self.assertEqual(built.kwargs, {"model": "d1.2-ultra", "screenshots": True,
                                        "max_steps": 4, "max_cost_usd": 0.05,
                                        "progress": None})

    def test_the_notes_file_is_where_a_running_loop_leaves_its_line(self):
        """``--progress`` is the only way the driver can speak while a step is still inside it.

        The file, not stdout: a browser inherits this process's console, and a caller that
        reads the report off stdout would have these lines mixed in with Chrome's own log.
        And each line is shut as it lands, because the caller unlinks the notes and takes the
        folder with it - on Windows a writer still holding the file leaves both undeletable.
        """
        outer = tempfile.TemporaryDirectory()
        self.addCleanup(outer.cleanup)
        lane = os.path.join(outer.name, "lane")
        os.mkdir(lane)
        path = os.path.join(lane, "notes.jsonl")
        note = self.build(self.ARGS + ["--progress", path]).kwargs["progress"]
        self.assertTrue(callable(note))
        note({"text": "Deciding step 1 of 4", "step": 1, "of": 4})
        note({"text": "Clicking Sign in", "step": 2, "of": 4})
        with open(path, encoding="utf-8") as written:
            lines = [json.loads(line) for line in written if line.strip()]
        self.assertEqual([line["text"] for line in lines],
                         ["Deciding step 1 of 4", "Clicking Sign in"])
        self.assertEqual(lines[1]["step"], 2)
        os.unlink(path)
        os.rmdir(lane)

    def build(self, argv):
        """Run ``main`` with the loop's constructor recorded, returning its call."""
        out = io.StringIO()
        with mock.patch.object(webagent, "WebTask", return_value=self.task(snapshots=1)) as built, \
                contextlib.redirect_stdout(out):
            web_driver.main(argv)
        return built.call_args

    def test_a_browser_log_on_stdout_cannot_hide_the_report(self):
        """The marker is what lets a caller find the JSON after whatever Chrome printed."""
        out = io.StringIO()
        with mock.patch.object(webagent, "WebTask", return_value=self.task(snapshots=1)):
            with contextlib.redirect_stdout(out):
                print("opening tab for https://shop.test/search")
                print("chrome: devtools listening on :9222")
                code = web_driver.main(self.ARGS)
        self.assertEqual(code, 0)
        prelude, _, payload = out.getvalue().partition(web_driver.SENTINEL)
        self.assertIn("opening tab", prelude)
        self.assertIn("devtools listening", prelude)
        self.assertEqual(json.loads(payload)["ok"], True)


class FakeTask:
    """A :class:`webagent.WebTask` with no browser and no bill.

    ``run`` yields the scripted number of snapshots, ``report`` returns the scripted
    account, and ``closed`` counts the hand-backs so a test can insist on exactly one.
    """

    def __init__(self, *, snapshots=1, report=None, run_raises=None, close_raises=None):
        self.snapshots = int(snapshots)
        self._report = dict(report or {})
        self._report.setdefault("ok", True)
        self._report.setdefault("engine", "liquid-d1.2")
        self._report.setdefault("steps", self.snapshots)
        self._report.setdefault("status", "done")
        self._report.setdefault("cost_usd", 0.0002)
        self.run_raises = run_raises
        self.close_raises = close_raises
        self.closed = 0

    def run(self):
        for _ in range(self.snapshots):
            if self.run_raises is not None:
                raise self.run_raises
            yield {"steps": 1}

    def report(self):
        return dict(self._report)

    def close(self):
        self.closed += 1
        if self.close_raises is not None:
            raise self.close_raises


if __name__ == "__main__":
    unittest.main()
