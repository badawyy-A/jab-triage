import pytest

from jab_triage.triage import sample_audit, triage
from tests.stub import ScriptedBackend
from tests.test_triage import PRESET, items

IDS = list(range(100))


def test_same_seed_same_sample():
    assert sample_audit(IDS, 0.1, seed=7) == sample_audit(IDS, 0.1, seed=7)


def test_different_seed_different_sample():
    assert sample_audit(IDS, 0.1, seed=7) != sample_audit(IDS, 0.1, seed=8)


def test_sample_keeps_input_order_and_is_a_subset():
    picked = sample_audit(IDS, 0.2, seed=1)
    assert picked == sorted(picked)
    assert set(picked) <= set(IDS)


@pytest.mark.parametrize(
    "n, rate, expected",
    [(100, 0.05, 5), (60, 0.05, 3), (19, 0.05, 1), (1, 0.05, 1), (10, 1.0, 10), (10, 0, 0)],
)
def test_sample_size_rounds_up(n, rate, expected):
    assert len(sample_audit(list(range(n)), rate, seed=0)) == expected


def test_empty_confident_gives_empty_audit():
    assert sample_audit([], 0.5, seed=0) == []


def test_triage_audits_only_confident_items():
    texts = [f"item {i}" for i in range(40)]
    script = {
        text: {"kind": ("bug", 0.9 if i % 2 else 0.1), "dup": (0.95,)}
        for i, text in enumerate(texts)
    }
    result = triage(ScriptedBackend(script), PRESET, items(*texts), audit_rate=0.25, seed=3)
    confident_ids = {r.id for r in result.confident}
    assert len(result.confident) == 20
    assert len(result.needs_review) == 20
    assert len(result.audit) == 5
    assert {r.id for r in result.audit} <= confident_ids
    assert result.summary["seed"] == 3
    assert result.summary["audit"] == 5
    assert result.summary["audit_rate"] == 0.25


def test_triage_without_seed_records_the_one_it_drew():
    texts = [f"item {i}" for i in range(10)]
    script = {text: {"kind": ("bug", 0.9), "dup": (0.95,)} for text in texts}
    first = triage(ScriptedBackend(script), PRESET, items(*texts), audit_rate=0.3)
    seed = first.summary["seed"]
    again = triage(ScriptedBackend(script), PRESET, items(*texts), audit_rate=0.3, seed=seed)
    assert [r.id for r in again.audit] == [r.id for r in first.audit]
