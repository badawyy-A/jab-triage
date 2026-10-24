import json

import pytest

from jab_triage import cli
from jab_triage.backends.base import BackendError
from tests.stub import ScriptedBackend
from tests.test_presets import VALID

PRESET = {
    **VALID,
    "name": "kinds",
    "state_template": "{text}",
    "questions": [VALID["questions"][0]],
}


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    presets_dir = tmp_path / "presets"
    presets_dir.mkdir()
    (presets_dir / "kinds.json").write_text(json.dumps(PRESET), encoding="utf-8")
    monkeypatch.setattr(cli, "PRESETS_DIR", presets_dir)
    for var in ("JEV_BACKEND", "OPENROUTER_API_KEY", "TYPESAFE_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    return tmp_path


def write_items(path, texts):
    path.write_text(json.dumps([{"id": i, "text": t} for i, t in enumerate(texts)]))
    return path


def stub_factory(script):
    backend = ScriptedBackend(script)
    return lambda env: backend


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_status_never_shows_the_key(workspace, monkeypatch, capsys):
    monkeypatch.setenv("JEV_BACKEND", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-very-secret")
    assert cli.main(["status"]) == 0
    out = capsys.readouterr().out
    assert "backend: openrouter" in out
    assert "OPENROUTER_API_KEY: set" in out
    assert "TYPESAFE_API_KEY: not set" in out
    assert "kinds: Example preset for tests." in out
    assert "very-secret" not in out


def test_presets_flags_invalid_files(workspace, capsys):
    (workspace / "presets" / "broken.json").write_text(json.dumps({**PRESET, "threshold": 2}))
    assert cli.main(["presets"]) == 1
    out = capsys.readouterr().out
    assert "kinds: Example preset" in out
    assert "broken: invalid:" in out


def test_presets_all_valid(workspace, capsys):
    assert cli.main(["presets"]) == 0


def test_triage_writes_all_four_files(workspace, capsys):
    texts = [f"issue {i}" for i in range(10)]
    script = {t: {"kind": ("bug", 0.9 if i < 8 else 0.2)} for i, t in enumerate(texts)}
    items = write_items(workspace / "items.json", texts)
    out = workspace / "out"
    args = ["triage", "--preset", "kinds", "--input", str(items), "--out", str(out), "--seed", "1"]
    assert cli.main(args, stub_factory(script)) == 0

    confident = read_jsonl(out / "confident.jsonl")
    needs_review = read_jsonl(out / "needs_review.jsonl")
    audit = read_jsonl(out / "audit.jsonl")
    summary = json.loads((out / "summary.json").read_text())
    assert len(confident) == 8
    assert len(needs_review) == 2
    assert len(audit) == 1
    assert audit[0]["id"] in {line["id"] for line in confident}
    # Confident lines leave the text out. Review and audit lines carry it, so the
    # agent does not have to open the input file.
    assert confident[0] == {
        "id": 0,
        "answers": {"kind": {"value": "bug", "confidence": 0.9}},
        "confidence": 0.9,
        "bucket": "confident",
    }
    assert needs_review[0]["text"] == "issue 8"
    assert audit[0]["text"] == texts[audit[0]["id"]]
    assert summary["seed"] == 1
    assert json.loads(capsys.readouterr().out) == summary


def test_ask_writes_one_results_file(workspace, capsys):
    items = write_items(workspace / "items.json", ["a", "b"])
    out = workspace / "results.jsonl"
    script = {"a": {"kind": ("bug", 0.9)}, "b": {"kind": ("feature", 0.5)}}
    args = ["ask", "--preset", "kinds", "--input", str(items), "--out", str(out)]
    assert cli.main(args, stub_factory(script)) == 0
    confident, review = read_jsonl(out)
    assert confident["bucket"] == "confident"
    assert "text" not in confident
    assert review["bucket"] == "needs_review"
    assert review["text"] == "b"


def test_preset_can_be_a_path(workspace, tmp_path, capsys):
    preset_path = tmp_path / "elsewhere.json"
    preset_path.write_text(json.dumps(PRESET))
    items = write_items(workspace / "items.json", ["a"])
    out = workspace / "results.jsonl"
    args = ["ask", "--preset", str(preset_path), "--input", str(items), "--out", str(out)]
    assert cli.main(args, stub_factory({"a": {"kind": ("bug", 0.9)}})) == 0


def test_missing_backend_is_a_clear_error(workspace, capsys):
    items = write_items(workspace / "items.json", ["a"])
    args = ["ask", "--preset", "kinds", "--input", str(items), "--out", str(workspace / "r")]
    assert cli.main(args) == 1
    assert "JEV_BACKEND is not set" in capsys.readouterr().err


def test_bad_input_is_a_clear_error(workspace, capsys):
    items = workspace / "items.json"
    items.write_text('{"not": "a list"}')
    args = ["ask", "--preset", "kinds", "--input", str(items), "--out", str(workspace / "r")]
    assert cli.main(args, stub_factory({})) == 1
    assert "JSON array" in capsys.readouterr().err


def test_unknown_preset_is_a_clear_error(workspace, capsys):
    items = write_items(workspace / "items.json", ["a"])
    args = ["ask", "--preset", "nope", "--input", str(items), "--out", str(workspace / "r")]
    assert cli.main(args, stub_factory({})) == 1
    assert "no preset named 'nope'" in capsys.readouterr().err


@pytest.mark.parametrize("flag, value", [("--threshold", "1.5"), ("--max-bytes", "0")])
def test_rejects_bad_numbers(workspace, flag, value):
    items = write_items(workspace / "items.json", ["a"])
    args = ["ask", "--preset", "kinds", "--input", str(items), "--out", "r", flag, value]
    with pytest.raises(SystemExit):
        cli.main(args, stub_factory({}))


def test_item_text_stays_out_of_output_without_debug(workspace, capsys):
    items = write_items(workspace / "items.json", ["private words", "more private words"])
    script = {"private words": BackendError("HTTP 500 from backend"), "more private words": {}}
    out = workspace / "out"
    args = ["triage", "--preset", "kinds", "--input", str(items), "--out", str(out)]
    assert cli.main(args, stub_factory(script)) == 0
    captured = capsys.readouterr()
    assert "HTTP 500" in captured.err
    assert "private" not in captured.out + captured.err
    assert "private" not in (out / "summary.json").read_text()
    assert "private" not in (out / "confident.jsonl").read_text()


def test_debug_logs_item_text(workspace, capsys):
    items = write_items(workspace / "items.json", ["private words"])
    args = ["ask", "--preset", "kinds", "--input", str(items), "--out", str(workspace / "r")]
    script = {"private words": {"kind": ("bug", 0.9)}}
    assert cli.main([*args, "--debug"], stub_factory(script)) == 0
    assert "private words" in capsys.readouterr().err
