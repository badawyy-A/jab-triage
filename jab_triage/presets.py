"""Load and validate presets.

A preset says how an item becomes the state sent to Jev, which questions to ask
about it, and the default confidence cutoff. Limits here come from the Jev API
notes in docs/jev-api-notes.md.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

QUESTION_TYPES = ("choice", "score", "noul")
MIN_OPTIONS = 2
MAX_CHOICE_OPTIONS = 255
MAX_SCORE_LEVELS = 10

PRESETS_DIR = Path(__file__).resolve().parent.parent / "presets"

_PRESET_KEYS = {"name", "description", "state_template", "questions", "threshold"}
_QUESTION_KEYS = {"id", "type", "prompt", "options"}
_NOUL_OPTION_KEYS = {"true", "false"}

# {text} or {meta.KEY}. Anything else in braces is rejected so typos fail early.
_PLACEHOLDER = re.compile(r"\{(text|meta\.[A-Za-z0-9_-]+)\}")
_ANY_BRACES = re.compile(r"\{([^{}]*)\}")


class PresetError(ValueError):
    pass


@dataclass(frozen=True)
class Question:
    id: str
    type: str
    prompt: str
    options: Any = None


@dataclass(frozen=True)
class Preset:
    name: str
    description: str
    state_template: Any
    questions: tuple[Question, ...]
    threshold: float


def load_preset(path: Path) -> Preset:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PresetError(f"{path}: cannot read preset: {exc}") from exc
    return parse_preset(data, source=str(path))


def find_preset(name_or_path: str, presets_dir: Path = PRESETS_DIR) -> Path:
    """Treat the argument as a file path if it exists, else as a preset name."""
    path = Path(name_or_path)
    if path.is_file():
        return path
    named = presets_dir / f"{name_or_path}.json"
    if named.is_file():
        return named
    raise PresetError(f"no preset named {name_or_path!r} in {presets_dir}")


def list_presets(presets_dir: Path = PRESETS_DIR) -> list[Path]:
    if not presets_dir.is_dir():
        return []
    return sorted(presets_dir.glob("*.json"))


def parse_preset(data: Any, source: str = "preset") -> Preset:
    if not isinstance(data, dict):
        raise PresetError(f"{source}: must be a JSON object")
    _check_keys(data, required=_PRESET_KEYS, allowed=_PRESET_KEYS, where=source)

    name = data["name"]
    if not isinstance(name, str) or not name.strip():
        raise PresetError(f"{source}: name must be a non-empty string")
    description = data["description"]
    if not isinstance(description, str):
        raise PresetError(f"{source}: description must be a string")

    threshold = data["threshold"]
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise PresetError(f"{source}: threshold must be a number")
    if not 0 <= threshold <= 1:
        raise PresetError(f"{source}: threshold must be between 0 and 1")

    template = data["state_template"]
    if not isinstance(template, (str, dict, list)):
        raise PresetError(f"{source}: state_template must be a string, object or array")
    _check_template(template, source)

    raw_questions = data["questions"]
    if not isinstance(raw_questions, list) or not raw_questions:
        raise PresetError(f"{source}: questions must be a non-empty list")
    questions = tuple(_parse_question(q, i, source) for i, q in enumerate(raw_questions))
    ids = [q.id for q in questions]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise PresetError(f"{source}: duplicate question ids: {', '.join(duplicates)}")

    return Preset(
        name=name,
        description=description,
        state_template=template,
        questions=questions,
        threshold=float(threshold),
    )


def render_state(template: Any, item: dict) -> Any:
    """Fill a state template from an item.

    A string that is exactly one placeholder becomes the raw value, so a list in
    meta stays a list in a JSON state. Placeholders inside longer strings are
    replaced with text. A missing meta key renders as null or an empty string.
    """
    if isinstance(template, dict):
        return {key: render_state(value, item) for key, value in template.items()}
    if isinstance(template, list):
        return [render_state(value, item) for value in template]
    if isinstance(template, str):
        whole = _PLACEHOLDER.fullmatch(template)
        if whole:
            return _lookup(whole.group(1), item)
        return _PLACEHOLDER.sub(lambda m: _as_text(_lookup(m.group(1), item)), template)
    return template


def _lookup(placeholder: str, item: dict) -> Any:
    if placeholder == "text":
        return item["text"]
    return (item.get("meta") or {}).get(placeholder.removeprefix("meta."))


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _check_template(template: Any, source: str) -> None:
    if isinstance(template, dict):
        for value in template.values():
            _check_template(value, source)
    elif isinstance(template, list):
        for value in template:
            _check_template(value, source)
    elif isinstance(template, str):
        for match in _ANY_BRACES.finditer(template):
            if not _PLACEHOLDER.fullmatch(match.group(0)):
                raise PresetError(
                    f"{source}: unknown placeholder {match.group(0)} in state_template; "
                    "use {text} or {meta.KEY}"
                )


def _parse_question(raw: Any, index: int, source: str) -> Question:
    where = f"{source}: questions[{index}]"
    if not isinstance(raw, dict):
        raise PresetError(f"{where}: must be an object")
    _check_keys(raw, required={"id", "type", "prompt"}, allowed=_QUESTION_KEYS, where=where)

    qid, qtype, prompt = raw["id"], raw["type"], raw["prompt"]
    if not isinstance(qid, str) or not qid.strip():
        raise PresetError(f"{where}: id must be a non-empty string")
    where = f"{source}: question {qid!r}"
    if qtype not in QUESTION_TYPES:
        raise PresetError(
            f"{where}: unknown type {qtype!r}; use one of {', '.join(QUESTION_TYPES)}"
        )
    if not isinstance(prompt, str) or not prompt.strip():
        raise PresetError(f"{where}: prompt must be a non-empty string")

    options = raw.get("options")
    if qtype == "choice":
        _check_choice_options(options, where)
    elif qtype == "score":
        _check_score_options(options, where)
    else:
        _check_noul_options(options, where)
    return Question(id=qid, type=qtype, prompt=prompt, options=options)


def _check_choice_options(options: Any, where: str) -> None:
    if not isinstance(options, dict):
        raise PresetError(f"{where}: choice options must be an object of name to description")
    if not MIN_OPTIONS <= len(options) <= MAX_CHOICE_OPTIONS:
        raise PresetError(
            f"{where}: choice needs {MIN_OPTIONS} to {MAX_CHOICE_OPTIONS} options, "
            f"got {len(options)}"
        )
    for name, description in options.items():
        if not name.strip():
            raise PresetError(f"{where}: option names must be non-empty")
        if not isinstance(description, (str, dict, list, type(None))):
            raise PresetError(f"{where}: option {name!r} has an invalid description")


def _check_score_options(options: Any, where: str) -> None:
    if not isinstance(options, list):
        raise PresetError(f"{where}: score options must be a list of levels, lowest first")
    if not MIN_OPTIONS <= len(options) <= MAX_SCORE_LEVELS:
        raise PresetError(
            f"{where}: score needs {MIN_OPTIONS} to {MAX_SCORE_LEVELS} levels, got {len(options)}"
        )
    for level in options:
        if not isinstance(level, (str, dict, list)):
            raise PresetError(f"{where}: each score level must be a string, object or array")


def _check_noul_options(options: Any, where: str) -> None:
    if options is None:
        return
    if not isinstance(options, dict) or not options:
        raise PresetError(f"{where}: noul options must be an object with true and/or false")
    unknown = set(options) - _NOUL_OPTION_KEYS
    if unknown:
        raise PresetError(f"{where}: noul options only take true and false, got {sorted(unknown)}")
    for key, description in options.items():
        if not isinstance(description, (str, dict, list, type(None))):
            raise PresetError(f"{where}: noul option {key!r} has an invalid description")


def _check_keys(data: dict, required: set[str], allowed: set[str], where: str) -> None:
    missing = sorted(required - set(data))
    if missing:
        raise PresetError(f"{where}: missing {', '.join(missing)}")
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise PresetError(f"{where}: unknown keys {', '.join(unknown)}")
