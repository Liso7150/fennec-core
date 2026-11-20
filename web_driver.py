"""Run Fennec's own browser loop as a child process, and report one JSON object.

:mod:`webagent` is the loop; this file is only its process boundary. It
exists because the loop drives Chrome through the browser-harness transport, which is not
a dependency of core itself: the driver is launched with whichever interpreter has that
package, and ``PYTHONPATH`` is aimed at this checkout so ``import webagent`` works
there too. Nothing in here reaches a third-party checkout.

The report keys are the ones the browser tool transcript already carries, so a run reads
the same whatever the page was; ``engine`` says which decider named the steps. A step can
sit inside one decision for tens of seconds, so ``--progress`` streams the loop's live notes
to a file while it runs instead of leaving the caller with nothing until the report.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

# The marker outlets.py looks for before its JSON. This runs in an interpreter chosen for
# the browser transport it has, not for this checkout, so it must not import from here to
# get the constant: the two copies are pinned together by a test.
SENTINEL = "@@@FENNEC-WEB-REPORT@@@"


def _emit(report: dict) -> None:
    """Print the report behind a marker, so a browser log on stdout cannot hide it."""
    sys.stdout.write(SENTINEL)
    sys.stdout.flush()
    json.dump(report, sys.stdout)
    sys.stdout.flush()


def _progress_writer(path):
    """Turn the loop's live notes into appended JSON lines, or None when nobody asked.

    The caller tails this file while the loop is inside a step, which is the only way a
    decision that takes half a minute can say so from a child process. Each line is opened,
    written and shut on its own, so a note is on disk the moment it is made and nothing here
    holds the file after that: the lane unlinks it when this process ends, and on Windows a
    writer that kept the handle open for the whole run makes that cleanup fail. A reader that
    catches a line mid-write skips it rather than this writer holding a lock: a clipped line
    is worth less than a stalled loop.
    """
    if not path:
        return None

    def report(event):
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(event) + "\n")

    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="start page of the task")
    parser.add_argument("--goal", required=True, help="end state to reach, not the hops")
    parser.add_argument("--max-steps", type=int, default=0, help="actions to allow at most")
    parser.add_argument("--timeout", type=float, default=240.0, help="wall-clock seconds")
    parser.add_argument("--model", default="", help="decision model (default: LIQUID_MODEL)")
    parser.add_argument("--max-cost-usd", type=float, default=0.0,
                        help="decision spend ceiling (default: FENNEC_WEB_MAX_COST)")
    parser.add_argument("--screenshots", action="store_true")
    parser.add_argument("--progress", default="",
                        help="append one JSON line per live status note to this file")
    args = parser.parse_args(argv)

    # Importing the loop is the availability test. The usual reason it fails is the
    # transport package, and saying so beats a traceback in the caller's error field.
    try:
        import webagent
    except ImportError as exc:
        _emit({"ok": False, "engine": "liquid-d1.2",
               "error": f"webagent is not importable: {exc}"})
        return 2

    limits = {}
    if args.max_steps:
        limits["max_steps"] = args.max_steps
    if args.max_cost_usd:
        limits["max_cost_usd"] = args.max_cost_usd
    # The first note is the tab opening, so the writer exists before the constructor runs.
    note = _progress_writer(args.progress)
    try:
        # Constructing opens the tab and takes the first read. A task that cannot be
        # authorised, or a page that will not load, has spent no decision by now.
        task = webagent.WebTask(args.url, args.goal, model=args.model, screenshots=bool(args.screenshots),
                                progress=note, **limits)
    except Exception as exc:
        _emit({"ok": False, "engine": "liquid-d1.2", "error": f"{type(exc).__name__}: {exc}"})
        return 3

    stopped = ""
    error = ""
    # The loop's own ceilings are steps and dollars; this one is the caller's, and it is
    # checked between steps because a step is the smallest unit that ends cleanly. Breaking
    # out here still prints the report, so a run cut short says what it cost and reached.
    deadline = time.monotonic() + args.timeout
    try:
        for _snapshot in task.run():
            if time.monotonic() > deadline:
                stopped = f"the {args.timeout:g}s wall clock"
                break
    except Exception as exc:
        # A browser that disappears mid-task leaves no page state to describe, so this is
        # an error rather than a blocked status. The spend so far is still the truth.
        error = f"{type(exc).__name__}: {exc}"

    report = task.report()
    try:
        task.close()
    except Exception as exc:
        # The answer stays the answer: a tab this process could not hand back is worth
        # naming, not worth discarding a finished, paid-for task over.
        report["leak"] = f"the tab could not be closed: {type(exc).__name__}: {exc}"
    if error:
        report.update({"ok": False, "error": error})
    if stopped:
        report["stopped_after"] = stopped
    _emit(report)
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
