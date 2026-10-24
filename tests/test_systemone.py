"""Tests for the HTTP backend. A fake transport stands in for the network, so
nothing here makes a real call."""

import json

import pytest

from jab_triage.backends.base import BackendError, FatalBackendError
from jab_triage.backends.systemone import (
    DEFAULT_MODEL,
    ENDPOINTS,
    SystemOneBackend,
    factory,
)
from jab_triage.presets import Question

KEY = "sk-test-very-secret"

QUESTIONS = (
    Question("kind", "choice", "Kind?", {"bug": "Broken", "feature": None}),
    Question("urgency", "score", "Urgent?", ["Can wait", "Today"]),
    Question("dup", "noul", "Duplicate?"),
)

ANSWERS = {
    "kind": {
        "type": "choice",
        "choice": "bug",
        "confidence": 0.75,
        "probabilities": {"bug": 0.875, "feature": 0.125},
    },
    "urgency": {
        "type": "score",
        "score": 0.9,
        "confidence": 0.8,
        "legend": {"0": "Can wait", "1": "Today"},
        "probabilities": {"0": 0.1, "1": 0.9},
    },
    "dup": {"type": "noul", "noul": 0.2},
}


def response(answers=ANSWERS, usage=None, model="typesafe/jev-1.14-20261014", **extra):
    usage = usage or {"input_tokens": 120, "output_tokens": 12, "cost": 0.000005}
    return json.dumps({"model": model, "answers": answers, "usage": usage, **extra}).encode()


class FakeTransport:
    """Returns the queued replies in order. A reply is (status, body) or an exception."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.requests = []

    def __call__(self, url, headers, body, timeout):
        self.requests.append({"url": url, "headers": headers, "body": json.loads(body)})
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        status, payload = reply
        return status, {}, payload


def backend(*replies, name="openrouter"):
    transport = FakeTransport(*replies)
    return SystemOneBackend(name, ENDPOINTS[name], KEY, transport=transport), transport


def test_builds_the_documented_request():
    b, transport = backend((200, response()))
    b.decide({"title": "T", "body": "B"}, QUESTIONS)
    [sent] = transport.requests
    assert sent["url"] == "https://openrouter.ai/api/v1/systemone"
    assert sent["headers"]["Authorization"] == f"Bearer {KEY}"
    assert sent["body"] == {
        "model": "jev-latest",
        "state": {"title": "T", "body": "B"},
        "questions": {
            "kind": {
                "type": "choice",
                "instructions": "Kind?",
                "criteria": {"bug": "Broken", "feature": None},
            },
            "urgency": {
                "type": "score",
                "instructions": "Urgent?",
                "criteria": ["Can wait", "Today"],
            },
            "dup": {"type": "noul", "instructions": "Duplicate?"},
        },
    }


def test_parses_every_answer_type():
    b, _ = backend((200, response()))
    decision = b.decide("state", QUESTIONS)
    assert decision.model == "typesafe/jev-1.14-20261014"
    kind, urgency, dup = (decision.answers[q] for q in ("kind", "urgency", "dup"))
    assert (kind.value, kind.confidence) == ("bug", 0.75)
    assert kind.probabilities == {"bug": 0.875, "feature": 0.125}
    assert (urgency.value, urgency.confidence) == (0.9, 0.8)
    assert dup.value == 0.2
    assert dup.confidence is None
    assert dup.probabilities == {"true": 0.2, "false": 0.8}


def test_reads_usage_and_cost():
    b, _ = backend((200, response()))
    usage = b.decide("state", QUESTIONS).usage
    assert (usage.input_tokens, usage.output_tokens, usage.cost) == (120, 12, 0.000005)


def test_cost_is_optional():
    b, _ = backend((200, response(usage={"input_tokens": 120, "output_tokens": 12})))
    assert b.decide("state", QUESTIONS).usage.cost is None


def test_missing_answer_is_left_out():
    b, _ = backend((200, response(answers={"kind": ANSWERS["kind"]})))
    assert set(b.decide("state", QUESTIONS).answers) == {"kind"}


@pytest.mark.parametrize(
    "answers",
    [
        {"kind": {**ANSWERS["kind"], "type": "score"}},
        {"kind": {**ANSWERS["kind"], "confidence": 1.5}},
        {"kind": {**ANSWERS["kind"], "choice": 3}},
        {"urgency": {**ANSWERS["urgency"], "score": "high"}},
        {"dup": {"type": "noul"}},
        {"dup": "yes"},
    ],
)
def test_malformed_answers_are_errors(answers):
    b, _ = backend((200, response(answers=answers)))
    with pytest.raises(BackendError, match="malformed"):
        b.decide("state", QUESTIONS)


@pytest.mark.parametrize(
    "payload",
    [b"not json", b"[]", json.dumps({"model": "m", "answers": {}}).encode()],
)
def test_malformed_responses_are_errors(payload):
    b, _ = backend((200, payload))
    with pytest.raises(BackendError):
        b.decide("state", QUESTIONS)


@pytest.mark.parametrize("status", [401, 402, 403, 404, 422])
def test_errors_that_would_repeat_stop_the_run(status):
    b, _ = backend((status, b"{}"))
    with pytest.raises(FatalBackendError, match=f"HTTP {status} from openrouter"):
        b.decide("state", QUESTIONS)


def test_rejected_key_names_the_variable_not_the_key():
    b, _ = backend((401, b"{}"))
    with pytest.raises(FatalBackendError) as exc:
        b.decide("state", QUESTIONS)
    assert "OPENROUTER_API_KEY" in str(exc.value)
    assert KEY not in str(exc.value)


def test_validation_error_shows_field_paths_but_not_the_input():
    body = json.dumps(
        {
            "detail": [
                {
                    "loc": ["body", "questions", "kind", "criteria"],
                    "msg": "bad",
                    "type": "x",
                    "input": "private item text",
                }
            ]
        }
    ).encode()
    b, _ = backend((422, body))
    with pytest.raises(FatalBackendError) as exc:
        b.decide("state", QUESTIONS)
    assert "questions.kind.criteria" in str(exc.value)
    assert "private" not in str(exc.value)


def test_context_overflow_has_a_clear_message():
    b, _ = backend((400, b'{"detail": "max_tokens_exceeded: private item text"}'))
    with pytest.raises(BackendError) as exc:
        b.decide("state", QUESTIONS)
    assert "over the model's context length" in str(exc.value)
    assert not isinstance(exc.value, FatalBackendError)
    assert "private" not in str(exc.value)


def test_other_errors_do_not_repeat_the_body():
    b, _ = backend((400, b'{"error": {"message": "private item text"}}'))
    with pytest.raises(BackendError) as exc:
        b.decide("state", QUESTIONS)
    assert str(exc.value) == "HTTP 400 from openrouter"


def test_nan_in_state_is_a_clear_error():
    b, transport = backend()
    with pytest.raises(BackendError, match="NaN"):
        b.decide({"x": float("nan")}, QUESTIONS)
    assert transport.requests == []


def test_repr_hides_the_key():
    b, _ = backend()
    assert KEY not in repr(b)


def test_factory_reads_key_and_model_from_env():
    built = factory("typesafe")({"TYPESAFE_API_KEY": KEY, "JEV_MODEL": "jev-1.14.0"})
    assert built.destination == "https://api.typesafe.ai/v1/systemone"
    assert built.model == "jev-1.14.0"
    assert factory("openrouter")({"OPENROUTER_API_KEY": KEY}).model == DEFAULT_MODEL


def test_factory_without_key_is_a_clear_error():
    with pytest.raises(BackendError, match="OPENROUTER_API_KEY is not set"):
        factory("openrouter")({})
