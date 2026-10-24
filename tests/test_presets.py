import copy
import json

import pytest

from jab_triage.presets import (
    MAX_CHOICE_OPTIONS,
    MAX_SCORE_LEVELS,
    PresetError,
    find_preset,
    list_presets,
    load_preset,
    parse_preset,
    render_state,
)

VALID = {
    "name": "example",
    "description": "Example preset for tests.",
    "state_template": {"title": "{meta.title}", "body": "{text}"},
    "threshold": 0.7,
    "questions": [
        {
            "id": "kind",
            "type": "choice",
            "prompt": "What kind of issue is this?",
            "options": {"bug": "Something is broken", "feature": None, "other": "Anything else"},
        },
        {
            "id": "urgency",
            "type": "score",
            "prompt": "How urgent is it?",
            "options": ["Can wait", "This week", "Today"],
        },
        {"id": "dup", "type": "noul", "prompt": "Does it say it is a duplicate?"},
    ],
}


def preset_with(**changes):
    data = copy.deepcopy(VALID)
    data.update(changes)
    return data


def question_with(index, **changes):
    data = copy.deepcopy(VALID)
    data["questions"][index].update(changes)
    return data


def test_valid_preset_parses():
    preset = parse_preset(VALID)
    assert preset.name == "example"
    assert preset.threshold == 0.7
    assert [q.id for q in preset.questions] == ["kind", "urgency", "dup"]
    assert preset.questions[2].options is None


@pytest.mark.parametrize("qtype", ["box", "text", "Choice", None])
def test_rejects_unknown_question_type(qtype):
    with pytest.raises(PresetError, match="unknown type"):
        parse_preset(question_with(0, type=qtype))


def test_choice_option_limit():
    options = {f"opt{i}": None for i in range(MAX_CHOICE_OPTIONS)}
    parse_preset(question_with(0, options=options))
    options["one_too_many"] = None
    with pytest.raises(PresetError, match="2 to 255 options"):
        parse_preset(question_with(0, options=options))


def test_choice_needs_two_options():
    with pytest.raises(PresetError, match="got 1"):
        parse_preset(question_with(0, options={"only": None}))


def test_choice_options_must_be_object():
    with pytest.raises(PresetError, match="must be an object"):
        parse_preset(question_with(0, options=["bug", "feature"]))


def test_score_level_limits():
    parse_preset(question_with(1, options=[f"level {i}" for i in range(MAX_SCORE_LEVELS)]))
    with pytest.raises(PresetError, match="2 to 10 levels"):
        parse_preset(question_with(1, options=[f"level {i}" for i in range(MAX_SCORE_LEVELS + 1)]))
    with pytest.raises(PresetError, match="got 1"):
        parse_preset(question_with(1, options=["only"]))


def test_noul_options_only_true_and_false():
    parse_preset(question_with(2, options={"true": "Says duplicate", "false": None}))
    with pytest.raises(PresetError, match="only take true and false"):
        parse_preset(question_with(2, options={"yes": "Says duplicate"}))


def test_rejects_duplicate_question_ids():
    data = copy.deepcopy(VALID)
    data["questions"][1]["id"] = "kind"
    with pytest.raises(PresetError, match="duplicate question ids: kind"):
        parse_preset(data)


@pytest.mark.parametrize("threshold", [-0.1, 1.5, "0.7", True])
def test_rejects_bad_threshold(threshold):
    with pytest.raises(PresetError, match="threshold"):
        parse_preset(preset_with(threshold=threshold))


def test_rejects_unknown_keys():
    with pytest.raises(PresetError, match="unknown keys model"):
        parse_preset(preset_with(model="jev-1.13"))
    with pytest.raises(PresetError, match="unknown keys criteria"):
        parse_preset(question_with(0, criteria={}))


def test_rejects_missing_keys():
    data = copy.deepcopy(VALID)
    del data["description"]
    with pytest.raises(PresetError, match="missing description"):
        parse_preset(data)


def test_rejects_empty_questions():
    with pytest.raises(PresetError, match="non-empty list"):
        parse_preset(preset_with(questions=[]))


def test_rejects_unknown_placeholder():
    with pytest.raises(PresetError, match=r"unknown placeholder \{body\}"):
        parse_preset(preset_with(state_template="Issue: {body}"))


def test_render_state_object_template():
    item = {"id": 1, "text": "It crashes", "meta": {"title": "Crash on start"}}
    assert render_state(VALID["state_template"], item) == {
        "title": "Crash on start",
        "body": "It crashes",
    }


def test_render_state_keeps_raw_values_and_handles_missing_meta():
    item = {"id": 1, "text": "x", "meta": {"labels": ["a", "b"]}}
    template = {"labels": "{meta.labels}", "line": "labels={meta.labels} owner={meta.owner}"}
    assert render_state(template, item) == {
        "labels": ["a", "b"],
        "line": 'labels=["a", "b"] owner=',
    }
    assert render_state("{meta.owner}", {"id": 1, "text": "x"}) is None


def test_load_and_find_preset(tmp_path):
    path = tmp_path / "example.json"
    path.write_text(json.dumps(VALID), encoding="utf-8")
    assert find_preset("example", tmp_path) == path
    assert find_preset(str(path), tmp_path / "elsewhere") == path
    assert load_preset(path).name == "example"
    assert list_presets(tmp_path) == [path]
    with pytest.raises(PresetError, match="no preset named"):
        find_preset("missing", tmp_path)


def test_load_preset_reports_bad_json(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(PresetError, match="cannot read preset"):
        load_preset(path)
