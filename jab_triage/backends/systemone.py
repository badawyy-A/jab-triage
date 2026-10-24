"""The System One HTTP API, which both OpenRouter and TypeSafe serve.

The two take the same request and return the same answers; OpenRouter adds a
cost to the usage. One class covers both, with a different URL and key. See
docs/jev-api-notes.md for where each field comes from.
"""

from __future__ import annotations

import http.client
import json
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from jab_triage.backends.base import (
    Answer,
    Backend,
    BackendError,
    Decision,
    FatalBackendError,
    Usage,
)
from jab_triage.presets import Question


@dataclass(frozen=True)
class Endpoint:
    url: str
    key_var: str


ENDPOINTS = {
    "openrouter": Endpoint("https://openrouter.ai/api/v1/systemone", "OPENROUTER_API_KEY"),
    "typesafe": Endpoint("https://api.typesafe.ai/v1/systemone", "TYPESAFE_API_KEY"),
}

# Both services accept this alias. OpenRouter maps it to ~typesafe/jev-latest.
# The model that actually answered is read from each response.
DEFAULT_MODEL = "jev-latest"
DEFAULT_TIMEOUT = 30.0

# (status, headers, body). Swapped out in tests so they never make a real call.
Transport = Callable[[str, dict[str, str], bytes, float], tuple[int, Mapping[str, str], bytes]]


def urllib_post(url: str, headers: dict[str, str], body: bytes, timeout: float):
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.headers, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.headers, exc.read()


def factory(name: str) -> Callable[[Mapping[str, str]], SystemOneBackend]:
    endpoint = ENDPOINTS[name]

    def build(env: Mapping[str, str]) -> SystemOneBackend:
        key = env.get(endpoint.key_var, "").strip()
        if not key:
            raise BackendError(f"{endpoint.key_var} is not set")
        model = env.get("JEV_MODEL", "").strip() or DEFAULT_MODEL
        return SystemOneBackend(name, endpoint, key, model)

    return build


class SystemOneBackend(Backend):
    def __init__(
        self,
        name: str,
        endpoint: Endpoint,
        api_key: str,
        model: str = DEFAULT_MODEL,
        timeout: float = DEFAULT_TIMEOUT,
        transport: Transport = urllib_post,
    ):
        self.name = name
        self.destination = endpoint.url
        self.model = model
        self.timeout = timeout
        self._key_var = endpoint.key_var
        self._api_key = api_key
        self._transport = transport

    def __repr__(self) -> str:
        # Spelled out so the key can never show up in a traceback or a log.
        return f"SystemOneBackend({self.name!r}, model={self.model!r})"

    def decide(self, state: Any, questions: tuple[Question, ...]) -> Decision:
        request = {
            "model": self.model,
            "state": state,
            "questions": {q.id: question_payload(q) for q in questions},
        }
        try:
            body = json.dumps(request, ensure_ascii=False, allow_nan=False).encode("utf-8")
        except ValueError as exc:
            raise BackendError(
                "state contains a number JSON cannot carry (NaN or infinity)"
            ) from exc
        return parse_response(self._post(body), questions)

    def _post(self, body: bytes) -> bytes:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        try:
            status, _, payload = self._transport(self.destination, headers, body, self.timeout)
        except (OSError, http.client.HTTPException) as exc:
            # The exception text can include the URL but never the key or body.
            raise BackendError(
                f"network error talking to {self.name} ({type(exc).__name__})"
            ) from exc
        if 200 <= status < 300:
            return payload
        raise self._status_error(status, payload)

    def _status_error(self, status: int, payload: bytes) -> BackendError:
        """Turn an error status into a message that never repeats the response body,
        since validation errors can quote the item text back."""
        where = f"HTTP {status} from {self.name}"
        if status == 401:
            return FatalBackendError(f"{where}: the key in {self._key_var} was rejected")
        if status == 402:
            return FatalBackendError(f"{where}: out of credits")
        if status == 403:
            return FatalBackendError(f"{where}: this key is not allowed to make this call")
        if status == 404:
            return FatalBackendError(f"{where}: check JEV_MODEL (sent {self.model!r})")
        if status == 422:
            # The same request shape goes out for every item, so this will not fix itself.
            return FatalBackendError(
                f"{where}: request failed validation at {_error_fields(payload)}"
            )
        if status == 400 and b"max_tokens_exceeded" in payload:
            return BackendError(f"{where}: item is over the model's context length")
        if status == 413:
            return BackendError(f"{where}: request body too large")
        return BackendError(where)


def question_payload(question: Question) -> dict:
    payload = {"type": question.type, "instructions": question.prompt}
    if question.options is not None:
        payload["criteria"] = question.options
    return payload


def parse_response(payload: bytes, questions: tuple[Question, ...]) -> Decision:
    try:
        data = json.loads(payload)
    except ValueError as exc:
        raise BackendError("response was not JSON") from exc
    if not isinstance(data, dict):
        raise BackendError("response was not a JSON object")
    raw_answers, raw_usage, model = data.get("answers"), data.get("usage"), data.get("model")
    if not isinstance(raw_answers, dict) or not isinstance(raw_usage, dict):
        raise BackendError("response is missing answers or usage")
    if not isinstance(model, str):
        raise BackendError("response is missing the model name")

    answers = {}
    for question in questions:
        raw = raw_answers.get(question.id)
        # A missing answer is left out here; triage reports it for that item.
        if raw is not None:
            answers[question.id] = parse_answer(question, raw)
    return Decision(answers=answers, usage=parse_usage(raw_usage), model=model)


def parse_answer(question: Question, raw: Any) -> Answer:
    where = f"{question.type} answer for question {question.id!r}"
    if not isinstance(raw, dict) or raw.get("type") != question.type:
        raise BackendError(f"malformed {where}")
    try:
        if question.type == "noul":
            p = _probability(raw["noul"])
            return Answer("noul", p, {"true": p, "false": 1 - p}, None)
        probabilities = {str(k): _probability(v) for k, v in raw["probabilities"].items()}
        confidence = _probability(raw["confidence"])
        if question.type == "choice":
            if not isinstance(raw["choice"], str):
                raise TypeError
            return Answer("choice", raw["choice"], probabilities, confidence)
        return Answer("score", _number(raw["score"]), probabilities, confidence)
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise BackendError(f"malformed {where}") from exc


def parse_usage(raw: dict) -> Usage:
    try:
        input_tokens, output_tokens = raw["input_tokens"], raw["output_tokens"]
        if not all(
            isinstance(n, int) and not isinstance(n, bool) for n in (input_tokens, output_tokens)
        ):
            raise TypeError
        cost = raw.get("cost")
        return Usage(input_tokens, output_tokens, None if cost is None else _number(cost))
    except (KeyError, TypeError) as exc:
        raise BackendError("malformed usage in response") from exc


def _number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("not a number")
    return float(value)


def _probability(value: Any) -> float:
    number = _number(value)
    if not 0 <= number <= 1:
        raise ValueError("not between 0 and 1")
    return number


def _error_fields(payload: bytes) -> str:
    """Field paths from a 422 body, like questions.kind.criteria. Only the paths:
    the rest of the body can contain a preview of the input."""
    try:
        detail = json.loads(payload)["detail"]
        paths = {".".join(str(part) for part in entry["loc"] if part != "body") for entry in detail}
    except (ValueError, KeyError, TypeError):
        return "an unknown field"
    return ", ".join(sorted(p for p in paths if p)) or "an unknown field"
