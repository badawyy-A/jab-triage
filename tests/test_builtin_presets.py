import json

import pytest

from jab_triage.presets import PRESETS_DIR, list_presets, load_preset, render_state

NAMES = ["ci-failure", "issue-triage", "search-relevance"]


def test_all_three_presets_are_there():
    assert [p.stem for p in list_presets(PRESETS_DIR)] == NAMES


@pytest.mark.parametrize("name", NAMES)
def test_preset_is_valid_and_named_after_its_file(name):
    preset = load_preset(PRESETS_DIR / f"{name}.json")
    assert preset.name == name


@pytest.mark.parametrize("name", NAMES)
def test_preset_renders_without_meta(name):
    preset = load_preset(PRESETS_DIR / f"{name}.json")
    state = render_state(preset.state_template, {"id": 1, "text": "some text"})
    assert "some text" in json.dumps(state)


@pytest.mark.parametrize("name", NAMES)
def test_choice_presets_end_with_a_catch_all(name):
    for question in load_preset(PRESETS_DIR / f"{name}.json").questions:
        if question.type == "choice":
            assert list(question.options)[-1] == "other"
