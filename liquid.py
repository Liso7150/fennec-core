"""Liquid d1.2: Fennec's own decision model, called directly.

A decision model is not a chat model. One forward pass over the state and the
question heads returns calibrated probabilities over the options *we* supplied,
with ``usage.output_tokens`` always 0. That is what makes it usable as a browser
policy: the set of legal actions is built by the executor from observed DOM
nodes, so the model can only pick an index that already exists. It cannot emit a
selector, a coordinate or a line of code, which removes the whole class of
prompt-injection-by-page-content that a generative actor has to defend against.

``d1.2`` and ``d1.2-ultra`` are hosted on the Liquid API (announced 2026-11-16);
there are no weights, so this call is the only way to use them and it is a paid
one. The wire format is the System One request the deprecated ``jev-ultrafast``
checkout spoke to, and the vendor states the request and response format carried
over unchanged from ``d1``/``d1.1``, so this module is a transport for a
documented contract rather than a guess at one:

* ``POST https://api.liquid.ai/decisions/v1/systemone``
* ``Authorization: Bearer $LIQUID_API_KEY``
* body ``{"model", "state", "questions"}``, answers keyed by question name

Two limits are enforced here before the request leaves, because both are cheaper to
fail on locally than to pay for: at most 128 questions per call, and the whole JSON
body under 4.5 MB. The per-question ceilings (65,536 tokens, 250,000 bytes of text)
are checked in bytes where bytes can approximate them and left to the API otherwise -
a client cannot count another vendor's tokens, and an answer that says so is more
honest than a heuristic that silently truncates the page text a decision was supposed
to be made over.

The same request goes to a local server when ``LIQUID_ENDPOINT`` names a loopback
address. ``LiquidAI/d1-2B-GGUF`` and the other open-weight d1 models are free under
the LFM Open License, and llama.cpp (>= 0.8.0) answers ``POST /v1/systemone`` with
byte-for-byte the envelope the hosted API uses - so a decision lane can run with no
key, no meter and no network. That lane takes ``d1-2B`` as the model name, and Noul,
Choice and Score as the only question types.

Requests carry structured ``state`` (the API documents plain text *or* a JSON
object) but flat string ``instructions``, which is the only form documented for
that field.
"""
from __future__ import annotations

import ipaddress
import json
import math
import os
import time
import urllib.error
import urllib.parse
import urllib.request

ENDPOINT = "https://api.liquid.ai/decisions/v1/systemone"
DEFAULT_MODEL = "d1.2"
# Dollars per million *input* tokens. Output is free on a decision model, so this
# is the whole price and the whole cost model.
PRICE_PER_MTOK = {"d1.2": 0.02, "d1.2-ultra": 0.042}
# The open-weight line, priced at what it actually costs: nothing. These are only
# accepted on a loopback endpoint, because a hosted request naming one of them would
# be billed and this table would be lying about the price.
LOCAL_MODELS = ("d1-2B", "d1-omni-2B", "d1-omni-600M")
LOCAL_MODEL = LOCAL_MODELS[0]
LOCAL_PRICE_PER_MTOK = dict.fromkeys(LOCAL_MODELS, 0.0)
PRICES = {**PRICE_PER_MTOK, **LOCAL_PRICE_PER_MTOK}
MODELS = tuple(PRICE_PER_MTOK)
ENDPOINT_ENV = "LIQUID_ENDPOINT"
MAX_QUESTIONS = 128
MAX_BODY_BYTES = 4_500_000
MAX_QUESTION_BYTES = 250_000
TIMEOUT_S = 25
RETRY_STATUS = (429, 529, 503)
ATTEMPTS = 3
RETRY_AFTER_CAP_S = 30.0
ERROR_BODY_CHARS = 300
# A Choice answer is a distribution, so it has to add up. The tolerance is not
# cosmetic: it is what catches a truncated or reordered answer.
PROBABILITY_SUM_TOLERANCE = 0.02


class LiquidError(RuntimeError):
    """The decision did not happen. Callers treat this as "no action executed"."""


def endpoint_for(environ=None) -> str:
    """Where a decision is asked for: the hosted API unless an endpoint is named."""
    return (os.environ if environ is None else environ).get(ENDPOINT_ENV, "").strip() or ENDPOINT


def is_local(url: str) -> bool:
    """Whether an endpoint is this machine's own System One server.

    Dropping the credential is only safe on a real loopback address, so this asks the
    IP layer rather than matching a prefix: ``127.0.0.1.evil.test`` starts like
    localhost and is not it.
    """
    host = (urllib.parse.urlparse(url).hostname or "").lower()
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def model_name(environ=None, url: str = "") -> str:
    env = os.environ if environ is None else environ
    named = env.get("LIQUID_MODEL", "").strip()
    if named:
        return named
    # A loopback endpoint serves an open-weight checkpoint, which the API never sees,
    # so naming the hosted default there would ask for a model that is not loaded.
    return LOCAL_MODEL if is_local(url or endpoint_for(env)) else DEFAULT_MODEL


def api_key(environ=None, url: str = "") -> str:
    env = os.environ if environ is None else environ
    if is_local(url or endpoint_for(env)):
        # Our own process, on our own loopback port. It checks no credential and bills
        # nothing, so demanding one here would block the one lane that costs nothing.
        return ""
    key = env.get("LIQUID_API_KEY", "").strip()
    if not key:
        raise LiquidError(
            "LIQUID_API_KEY is not set, so the d1.2 decision model cannot be called. "
            "A key comes from the Liquid console; requests are billed by input token "
            f"(${PRICE_PER_MTOK[DEFAULT_MODEL]:.3f} per million on {DEFAULT_MODEL}). "
            f"To run the free open-weight {LOCAL_MODEL} instead, set {ENDPOINT_ENV} to a "
            f"loopback System One server, for example http://127.0.0.1:8199/v1/systemone."
        )
    return key


def cost_usd(model: str, usage: dict) -> float:
    """Billed cost from the usage the API returned, never from an estimate."""
    tokens = (usage or {}).get("input_tokens") or 0
    return round(tokens / 1_000_000 * PRICES.get(model, PRICE_PER_MTOK[DEFAULT_MODEL]), 8)


def _request_bytes(body: dict) -> bytes:
    """Serialise once, so the size checked is the size sent.

    ``ensure_ascii=False`` matters here as well as on the wire: a page with CJK
    text costs the API its 250,000-byte question ceiling in characters it reads as
    three bytes each, and escaping them to \\uXXXX would inflate that same text a
    further sixfold here while buying nothing.
    """
    return json.dumps(body, ensure_ascii=False).encode("utf-8")


def _retry_after(headers) -> float:
    raw = (headers.get("Retry-After") or "").strip() if headers else ""
    try:
        seconds = float(raw)
    except ValueError:
        return 0.0
    return max(0.0, min(seconds, RETRY_AFTER_CAP_S))


def post_json(body: dict, *, key: str, endpoint: str = ENDPOINT,
              timeout: float = TIMEOUT_S, opener=None, sleep=time.sleep) -> dict:
    """One System One call, with the retries a rate-limited decision loop needs.

    Retries cover only the statuses that mean "not now": 429 plus the two
    overload codes. A 4xx about our own payload is returned immediately, because
    repeating it would bill the same mistake three times. No retry ever follows a
    response we could not parse, so a caller cannot double-charge an ambiguous
    request by mistake - if the connection dies the error says so and says no
    decision was made.

    The key is only ever placed in a header. Every message raised here is built
    from the status code and the API's own ``error.type``/``error.code``/
    ``error.message``, so a credential cannot ride out in a transcript.
    """
    payload = _request_bytes(body)
    if len(payload) > MAX_BODY_BYTES:
        raise LiquidError(
            f"the decision request is {len(payload):,} bytes against the documented "
            f"{MAX_BODY_BYTES:,}-byte ceiling; narrow the page text or the question set "
            "instead of sending it and hoping")
    headers = {"content-type": "application/json"}
    if key:
        headers["authorization"] = f"Bearer {key}"
    if opener is None:
        # A loopback server must not be reached through this machine's proxy: the request
        # would go to a box that has no business answering for it, and the failure would
        # read "d1.2 could not be reached".
        openers = (urllib.request.build_opener(urllib.request.ProxyHandler({}))
                   if is_local(endpoint) else urllib.request.build_opener())
    else:
        openers = opener
    last = None
    for attempt in range(ATTEMPTS):
        request = urllib.request.Request(endpoint, data=payload, headers=headers, method="POST")
        try:
            with openers.open(request, timeout=timeout) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", "replace")
            except Exception:  # an unreadable error body must not hide the status
                pass
            message = ""
            try:
                parsed = json.loads(detail)
                error = parsed.get("error") or {}
                message = " ".join(
                    part for part in (str(error.get("type") or ""), str(error.get("code") or ""),
                                      str(error.get("message") or "")[:ERROR_BODY_CHARS]) if part.strip())
            except (ValueError, AttributeError):
                message = detail[:ERROR_BODY_CHARS].strip()
            if exc.code in RETRY_STATUS and attempt < ATTEMPTS - 1:
                last = LiquidError(
                    f"d1.2 returned HTTP {exc.code} {message}".strip() + "; retrying")
                sleep(_retry_after(exc.headers) or 0.5 * 2 ** attempt)
                continue
            raise LiquidError(
                f"d1.2 returned HTTP {exc.code} {message}; no decision was made".strip()) from None
        except (urllib.error.URLError, OSError):
            raise LiquidError(
                "d1.2 could not be reached; no decision was made") from None
        try:
            answer = json.loads(raw.decode("utf-8", "replace"))
        except ValueError:
            raise LiquidError("d1.2 answered with something that is not JSON; no decision was made") from None
        if not isinstance(answer, dict) or "answers" not in answer:
            raise LiquidError("d1.2 answered without an answers object; no decision was made")
        return answer
    raise last or LiquidError("d1.2 is unavailable; no decision was made")


def validate_choice(answer: dict, ids) -> dict:
    """Accept a Choice answer only if it is a distribution over exactly ``ids``.

    The chosen option must be present, every id must be scored, the scores must
    sum to one, and the choice must be the argmax. All four are load-bearing: a
    policy that trusts ``choice`` without checking it against ``probabilities``
    would act on a key the model invented, and an answer that skipped options is
    indistinguishable from one that ranked them at zero.
    """
    wanted = set(ids)
    try:
        probabilities = answer["probabilities"]
        valid = (
            isinstance(probabilities, dict)
            and answer.get("choice") in wanted
            and set(probabilities) == wanted
            and all(type(n) in (int, float) and math.isfinite(n) and 0 <= n <= 1
                    for n in [*probabilities.values(), answer["confidence"]])
            and abs(sum(probabilities.values()) - 1) < PROBABILITY_SUM_TOLERANCE
            and probabilities[answer["choice"]] >= max(probabilities.values()) - 1e-6
        )
    except (KeyError, TypeError, ValueError):
        valid = False
    if not valid:
        raise LiquidError(
            "d1.2 returned a choice that is not a distribution over the offered options; "
            "no action executed")
    return answer


def validate_noul(answer: dict) -> float:
    try:
        value = answer["noul"]
    except (KeyError, TypeError):
        raise LiquidError("d1.2 returned no noul probability; no action executed") from None
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise LiquidError("d1.2 returned an out-of-range noul probability; no action executed")
    return float(value)


def check_questions(questions: dict) -> dict:
    """Reject a malformed question set locally, before it is billed.

    ``criteria`` is a mapping for a Choice (option id -> description) and a
    sequence for a Score (level descriptions, in order). Getting that backwards is
    the most likely mistake a caller can make, and the API's complaint about it
    would arrive after the request had already been sent.
    """
    if not isinstance(questions, dict) or not questions:
        raise LiquidError("a decision needs at least one named question")
    if len(questions) > MAX_QUESTIONS:
        raise LiquidError(
            f"{len(questions)} questions in one call exceeds the documented limit of "
            f"{MAX_QUESTIONS}; split the decision or narrow the option sets")
    for name, question in questions.items():
        if not isinstance(question, dict) or not question.get("type"):
            raise LiquidError(f"question {name!r} has no type")
        kind = question["type"]
        criteria = question.get("criteria")
        if kind == "choice" and not isinstance(criteria, dict):
            raise LiquidError(f"choice question {name!r} needs criteria as an option->description mapping")
        if kind == "score" and not isinstance(criteria, (list, tuple)):
            raise LiquidError(f"score question {name!r} needs criteria as an ordered list of levels")
        if criteria is not None and kind not in ("choice", "score", "multiple"):
            raise LiquidError(f"question {name!r} is type {kind!r} and cannot carry criteria")
        if isinstance(question.get("instructions"), str) and \
                len(question["instructions"].encode("utf-8")) > MAX_QUESTION_BYTES:
            raise LiquidError(
                f"question {name!r} sends more instructions than the documented "
                f"{MAX_QUESTION_BYTES:,}-byte per-question ceiling")
    return questions


def decide(state, questions: dict, *, model: str = "", key: str = "",
           endpoint: str = "", timeout: float = TIMEOUT_S,
           opener=None, sleep=time.sleep, environ=None) -> dict:
    """Ask every head of one decision in a single call.

    Returns the answers keyed by question name, plus the model that answered, the
    billed usage, the cost that usage implies and the round-trip latency, so the
    caller can account for what a step cost instead of finding out from an invoice.
    """
    check_questions(questions)
    server = endpoint or endpoint_for(environ)
    local = is_local(server)
    chosen = model or model_name(environ, server)
    wanted = LOCAL_MODELS if local else MODELS
    if chosen not in wanted:
        raise LiquidError(
            f"{chosen!r} is not a {'servable' if local else 'hosted'} decision model; "
            + (f"the open-weight line is {' or '.join(LOCAL_MODELS)}, free to run locally"
               if local else
               "the d1.2 line is " + " or ".join(f"{name} (${price:.3f}/M)"
                                                 for name, price in PRICE_PER_MTOK.items())))
    body = {"model": chosen, "state": state, "questions": questions}
    started = time.perf_counter()
    result = post_json(body, key=key or api_key(environ, server), endpoint=server,
                       timeout=timeout, opener=opener, sleep=sleep)
    usage = result.get("usage") or {}
    return {
        "answers": result["answers"],
        "model": result.get("model") or chosen,
        "usage": usage,
        "cost_usd": cost_usd(chosen, usage),
        "latency_ms": round((time.perf_counter() - started) * 1000),
        "request_bytes": len(_request_bytes(body)),
    }
