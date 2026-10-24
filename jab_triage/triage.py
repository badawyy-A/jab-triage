"""Ask Jev about each item and sort the items by how sure it was.

Each item gets one call with all of the preset's questions. Its confidence is
the lowest confidence across those questions, so one unsure answer is enough to
send it to review. Items at or above the threshold are confident.
"""

from __future__ import annotations

import json
import logging
import math
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from jab_triage.backends.base import Answer, Backend, BackendError, Decision, FatalBackendError
from jab_triage.cost import CostTally
from jab_triage.presets import Preset, render_state

CONFIDENT = "confident"
NEEDS_REVIEW = "needs_review"

# Above roughly this many bytes of text, OpenRouter sends Jev requests to the
# slower, less accurate long-context mode. See docs/jev-api-notes.md.
DEFAULT_MAX_STATE_BYTES = 32_000
DEFAULT_AUDIT_RATE = 0.05
DEFAULT_CONCURRENCY = 8

_ITEM_KEYS = {"id", "text", "meta"}

log = logging.getLogger(__name__)


class InputError(ValueError):
    pass


@dataclass
class ItemResult:
    item: dict
    answers: dict[str, dict] | None
    confidence: float | None
    bucket: str
    error: str | None = None

    @property
    def id(self) -> Any:
        return self.item["id"]

    def to_json(self, with_text: bool = False) -> dict:
        """One output line. with_text adds the item's text and meta, for lines the
        agent has to read anyway, so it never needs to open the whole input file."""
        line = {
            "id": self.id,
            "answers": self.answers,
            "confidence": self.confidence,
            "bucket": self.bucket,
        }
        if self.error is not None:
            line["error"] = self.error
        if with_text:
            line["text"] = self.item["text"]
            if self.item.get("meta"):
                line["meta"] = self.item["meta"]
        return line


@dataclass
class RunResult:
    preset: str
    backend: str
    threshold: float
    max_state_bytes: int
    items: list[ItemResult]
    usage: CostTally
    models: set[str]
    concurrency: int
    seconds: float = 0.0

    def in_bucket(self, bucket: str) -> list[ItemResult]:
        return [r for r in self.items if r.bucket == bucket]

    def summary(self) -> dict:
        return {
            "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "preset": self.preset,
            "backend": self.backend,
            "models": sorted(self.models),
            "threshold": self.threshold,
            "max_state_bytes": self.max_state_bytes,
            "concurrency": self.concurrency,
            "seconds": self.seconds,
            "items": len(self.items),
            CONFIDENT: len(self.in_bucket(CONFIDENT)),
            NEEDS_REVIEW: len(self.in_bucket(NEEDS_REVIEW)),
            "errors": sum(1 for r in self.items if r.error is not None),
            "usage": self.usage.to_json(),
        }


def parse_items(data: Any) -> list[dict]:
    """Check the input is a JSON array of {id, text, meta?} with unique ids.

    Error messages name the item by position and id, never by its text.
    """
    if not isinstance(data, list):
        raise InputError("input must be a JSON array of {id, text, meta?}")
    seen = set()
    for index, item in enumerate(data):
        where = f"item {index}"
        if not isinstance(item, dict):
            raise InputError(f"{where}: must be an object")
        unknown = sorted(set(item) - _ITEM_KEYS)
        if unknown:
            raise InputError(f"{where}: unknown keys {', '.join(unknown)}")
        item_id = item.get("id")
        if isinstance(item_id, bool) or not isinstance(item_id, (str, int)):
            raise InputError(f"{where}: id must be a string or an integer")
        where = f"item {index} (id {item_id!r})"
        if item_id in seen:
            raise InputError(f"{where}: duplicate id")
        seen.add(item_id)
        if not isinstance(item.get("text"), str):
            raise InputError(f"{where}: text must be a string")
        if not isinstance(item.get("meta", {}), dict):
            raise InputError(f"{where}: meta must be an object")
    return data


def answer_confidence(answer: Answer) -> float:
    """The API's confidence if it gives one, else the top probability.

    Rounded to four places before it is compared with the threshold, so the
    number in the output is the one the bucket was decided on, and float noise
    like 1 - 0.33 = 0.6699999999999999 does not end up in what the agent reads.
    """
    if answer.confidence is not None:
        return round(answer.confidence, 4)
    return round(max(answer.probabilities.values()), 4)


def run(
    backend: Backend,
    preset: Preset,
    items: list[dict],
    threshold: float | None = None,
    max_state_bytes: int = DEFAULT_MAX_STATE_BYTES,
    concurrency: int = DEFAULT_CONCURRENCY,
    debug: bool = False,
) -> RunResult:
    """Ask about every item, several at a time. Results keep the input order.

    A failed item goes to review and the rest carry on. A FatalBackendError, like
    a rejected key, stops the run and is raised; items not yet sent never are.
    """
    if threshold is None:
        threshold = preset.threshold
    result = RunResult(
        preset=preset.name,
        backend=backend.name,
        threshold=threshold,
        max_state_bytes=max_state_bytes,
        items=[],
        usage=CostTally(),
        models=set(),
        concurrency=concurrency,
    )
    stop = threading.Event()

    def ask(item: dict) -> tuple[ItemResult, Decision | None] | None:
        if stop.is_set():
            return None
        try:
            return ask_one(backend, preset, item, threshold, max_state_bytes, debug)
        except FatalBackendError:
            stop.set()
            raise

    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(ask, item) for item in items]
        try:
            # The pool starts items in order, so every item before a fatal one was
            # started before the stop and has a real result.
            for item, future in zip(items, futures, strict=True):
                item_result, decision = future.result()
                if decision is not None:
                    result.usage.add(decision.usage)
                    result.models.add(decision.model)
                if item_result.error is not None:
                    log.warning("item %r: %s", item["id"], item_result.error)
                result.items.append(item_result)
        except FatalBackendError:
            for pending in futures:
                pending.cancel()
            raise
    result.seconds = round(time.monotonic() - started, 3)
    return result


def ask_one(
    backend: Backend,
    preset: Preset,
    item: dict,
    threshold: float,
    max_state_bytes: int,
    debug: bool = False,
) -> tuple[ItemResult, Decision | None]:
    """Ask about one item. Returns the decision too, if a call was made, for its usage."""
    state = render_state(preset.state_template, item)
    size = _state_bytes(state)
    if size > max_state_bytes:
        return _failed(
            item, f"state is {size} bytes, over the {max_state_bytes} limit; not sent"
        ), None
    if debug:
        log.debug("item %r state: %s", item["id"], json.dumps(state, ensure_ascii=False))

    try:
        decision = backend.decide(state, preset.questions)
    except FatalBackendError:
        # Every later item would fail the same way, so stop instead of trying them all.
        raise
    except BackendError as exc:
        return _failed(item, str(exc)), None
    except Exception as exc:
        # The message of an unexpected error could include item text, so only its type is kept.
        return _failed(item, f"unexpected {type(exc).__name__} from backend"), None

    answers = {}
    for question in preset.questions:
        answer = decision.answers.get(question.id)
        if answer is None:
            return _failed(item, f"no answer for question {question.id!r}"), decision
        if answer.confidence is None and not answer.probabilities:
            return _failed(item, f"no confidence for question {question.id!r}"), decision
        answers[question.id] = {"value": answer.value, "confidence": answer_confidence(answer)}

    confidence = min(a["confidence"] for a in answers.values())
    bucket = CONFIDENT if confidence >= threshold else NEEDS_REVIEW
    return ItemResult(item, answers, confidence, bucket), decision


@dataclass
class TriageResult:
    confident: list[ItemResult]
    needs_review: list[ItemResult]
    # A random sample of confident items, for the agent to check Jev against.
    audit: list[ItemResult]
    summary: dict


def triage(
    backend: Backend,
    preset: Preset,
    items: list[dict],
    threshold: float | None = None,
    audit_rate: float = DEFAULT_AUDIT_RATE,
    seed: int | None = None,
    max_state_bytes: int = DEFAULT_MAX_STATE_BYTES,
    concurrency: int = DEFAULT_CONCURRENCY,
    debug: bool = False,
) -> TriageResult:
    """Run every item, then sample confident ones for audit.

    Without a seed one is drawn at random. It goes in the summary either way, so
    any run's audit sample can be reproduced.
    """
    if seed is None:
        seed = random.SystemRandom().randrange(2**32)
    result = run(
        backend,
        preset,
        items,
        threshold=threshold,
        max_state_bytes=max_state_bytes,
        concurrency=concurrency,
        debug=debug,
    )
    confident = result.in_bucket(CONFIDENT)
    audit = sample_audit(confident, audit_rate, seed)
    summary = result.summary()
    summary.update({"audit_rate": audit_rate, "seed": seed, "audit": len(audit)})
    return TriageResult(
        confident=confident,
        needs_review=result.in_bucket(NEEDS_REVIEW),
        audit=audit,
        summary=summary,
    )


def sample_audit(confident: list, rate: float, seed: int) -> list:
    """Pick rate * len(confident) items, rounded up, keeping their input order.

    Rounding up means any non-zero rate audits at least one item.
    """
    if not confident or rate <= 0:
        return []
    # Round before ceil so that 100 * 0.07, which is 7.000000000000001 in floats, gives 7.
    k = min(len(confident), math.ceil(round(len(confident) * rate, 9)))
    chosen = sorted(random.Random(seed).sample(range(len(confident)), k))
    return [confident[i] for i in chosen]


def _failed(item: dict, error: str) -> ItemResult:
    return ItemResult(item, None, None, NEEDS_REVIEW, error)


def _state_bytes(state: Any) -> int:
    text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
    return len(text.encode("utf-8"))
