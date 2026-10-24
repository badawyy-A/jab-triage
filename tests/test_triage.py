import logging

import pytest

from jab_triage.backends.base import BackendError, FatalBackendError
from jab_triage.presets import parse_preset
from jab_triage.triage import (
    CONFIDENT,
    NEEDS_REVIEW,
    InputError,
    parse_items,
    run,
)
from tests.stub import ScriptedBackend

PRESET = parse_preset(
    {
        "name": "test",
        "description": "Kind and duplicate check.",
        "state_template": "{text}",
        "threshold": 0.7,
        "questions": [
            {
                "id": "kind",
                "type": "choice",
                "prompt": "Kind?",
                "options": {"bug": None, "feature": None},
            },
            {"id": "dup", "type": "noul", "prompt": "Duplicate?"},
        ],
    }
)


def items(*texts):
    return [{"id": i, "text": text} for i, text in enumerate(texts)]


def run_script(script, texts, **kwargs):
    backend = ScriptedBackend(script)
    return backend, run(backend, PRESET, items(*texts), **kwargs)


def test_exactly_at_threshold_is_confident():
    _, result = run_script({"a": {"kind": ("bug", 0.7), "dup": (0.95,)}}, ["a"])
    assert result.items[0].confidence == 0.7
    assert result.items[0].bucket == CONFIDENT


def test_just_below_threshold_needs_review():
    _, result = run_script({"a": {"kind": ("bug", 0.6999), "dup": (0.95,)}}, ["a"])
    assert result.items[0].bucket == NEEDS_REVIEW


def test_item_confidence_is_the_lowest_question():
    _, result = run_script({"a": {"kind": ("bug", 0.99), "dup": (0.4,)}}, ["a"])
    [item] = result.items
    # noul has no API confidence, so the top probability max(0.4, 0.6) is used.
    assert item.answers["dup"] == {"value": 0.4, "confidence": 0.6}
    assert item.confidence == 0.6
    assert item.bucket == NEEDS_REVIEW


def test_threshold_override():
    _, result = run_script({"a": {"kind": ("bug", 0.6), "dup": (0.9,)}}, ["a"], threshold=0.5)
    assert result.threshold == 0.5
    assert result.items[0].bucket == CONFIDENT


def test_empty_input():
    backend, result = run_script({}, [])
    assert result.items == []
    assert backend.calls == []
    summary = result.summary()
    assert summary["items"] == 0
    assert summary[CONFIDENT] == 0
    assert summary["usage"]["calls"] == 0


def test_item_over_the_limit_is_not_sent():
    script = {"short": {"kind": ("bug", 0.9), "dup": (0.9,)}}
    backend, result = run_script(script, ["short", "x" * 11], max_state_bytes=10)
    assert backend.calls == ["short"]
    long_item = result.items[1]
    assert long_item.bucket == NEEDS_REVIEW
    assert long_item.answers is None
    assert "11 bytes, over the 10 limit" in long_item.error


def test_limit_counts_utf8_bytes():
    # Six characters, twelve bytes.
    _, result = run_script({}, ["éééééé"], max_state_bytes=10)
    assert "12 bytes" in result.items[0].error


def test_backend_error_on_one_item_keeps_the_others():
    script = {
        "a": {"kind": ("bug", 0.9), "dup": (0.9,)},
        "b": BackendError("HTTP 500 from backend"),
        "c": {"kind": ("feature", 0.8), "dup": (0.1,)},
    }
    _, result = run_script(script, ["a", "b", "c"])
    assert [r.bucket for r in result.items] == [CONFIDENT, NEEDS_REVIEW, CONFIDENT]
    assert result.items[1].error == "HTTP 500 from backend"
    summary = result.summary()
    assert summary["errors"] == 1
    assert summary["usage"]["calls"] == 2


def test_unexpected_error_message_is_not_kept():
    script = {"secret text": RuntimeError("failed on secret text")}
    _, result = run_script(script, ["secret text"])
    assert result.items[0].error == "unexpected RuntimeError from backend"


def test_missing_answer_sends_item_to_review_but_counts_usage():
    _, result = run_script({"a": {"kind": ("bug", 0.9)}}, ["a"])
    assert result.items[0].error == "no answer for question 'dup'"
    assert result.usage.calls == 1


def test_summary_records_models_and_usage():
    script = {"a": {"kind": ("bug", 0.9), "dup": (0.9,)}}
    _, result = run_script(script, ["a"])
    summary = result.summary()
    assert summary["models"] == ["stub-model"]
    assert summary["preset"] == "test"
    assert summary["backend"] == "stub"
    assert summary["usage"]["input_tokens"] == 100


def test_item_text_is_logged_only_with_debug(caplog):
    script = {"private words": {"kind": ("bug", 0.9), "dup": (0.9,)}}
    caplog.set_level(logging.DEBUG)
    run_script(script, ["private words"])
    assert "private words" not in caplog.text
    run_script(script, ["private words"], debug=True)
    assert "private words" in caplog.text


def test_failure_log_names_id_not_text(caplog):
    script = {"private words": BackendError("HTTP 429 from backend")}
    run_script(script, ["private words"])
    assert "item 0: HTTP 429" in caplog.text
    assert "private words" not in caplog.text


def test_parse_items_accepts_valid_input():
    data = [{"id": "a", "text": "x", "meta": {"k": 1}}, {"id": 2, "text": "y"}]
    assert parse_items(data) == data


@pytest.mark.parametrize(
    "data, message",
    [
        ({"id": 1}, "JSON array"),
        (["x"], "must be an object"),
        ([{"id": 1, "text": "x", "body": "y"}], "unknown keys body"),
        ([{"text": "x"}], "id must be"),
        ([{"id": True, "text": "x"}], "id must be"),
        ([{"id": 1, "text": 5}], "text must be a string"),
        ([{"id": 1, "text": "x", "meta": []}], "meta must be an object"),
        ([{"id": 1, "text": "x"}, {"id": 1, "text": "y"}], "duplicate id"),
    ],
)
def test_parse_items_rejects_bad_input(data, message):
    with pytest.raises(InputError, match=message):
        parse_items(data)


def test_parse_items_error_does_not_include_text():
    with pytest.raises(InputError) as exc:
        parse_items([{"id": 1, "text": "private words"}, {"id": 1, "text": "more private"}])
    assert "private" not in str(exc.value)


def test_to_json_with_text_adds_text_and_meta():
    data = [{"id": "a", "text": "x", "meta": {"title": "T"}}, {"id": "b", "text": "y"}]
    backend = ScriptedBackend({"x": {"kind": ("bug", 0.9), "dup": (0.9,)}, "y": {}})
    with_meta, without_meta = run(backend, PRESET, data).items
    assert "text" not in with_meta.to_json()
    assert with_meta.to_json(with_text=True)["meta"] == {"title": "T"}
    assert without_meta.to_json(with_text=True)["text"] == "y"
    assert "meta" not in without_meta.to_json(with_text=True)


def test_confidence_is_rounded_before_bucketing():
    # 1 - 0.30001 is 0.69999 in exact terms; rounded to four places it is 0.7.
    _, result = run_script({"a": {"kind": ("bug", 0.9), "dup": (0.30001,)}}, ["a"])
    assert result.items[0].confidence == 0.7
    assert result.items[0].bucket == CONFIDENT
    # 1 - 0.33 is 0.6699999999999999 in floats.
    _, result = run_script({"a": {"kind": ("bug", 0.9), "dup": (0.33,)}}, ["a"])
    assert result.items[0].answers["dup"]["confidence"] == 0.67


def test_fatal_error_stops_the_run():
    script = {
        "a": {"kind": ("bug", 0.9), "dup": (0.9,)},
        "b": FatalBackendError("HTTP 401 from stub: the key was rejected"),
        "c": {"kind": ("bug", 0.9), "dup": (0.9,)},
    }
    backend = ScriptedBackend(script)
    with pytest.raises(FatalBackendError, match="401"):
        run(backend, PRESET, items("a", "b", "c"), concurrency=1)
    assert "c" not in backend.calls
