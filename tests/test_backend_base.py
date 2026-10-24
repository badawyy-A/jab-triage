import pytest

from jab_triage.backends.base import Backend
from jab_triage.presets import Question
from tests.stub import ScriptedBackend


def test_backend_is_abstract():
    with pytest.raises(TypeError):
        Backend()


def test_stub_follows_the_interface():
    questions = (
        Question("kind", "choice", "Kind?", {"bug": None, "other": None}),
        Question("dup", "noul", "Duplicate?"),
    )
    backend = ScriptedBackend({"state": {"kind": ("bug", 0.9), "dup": (0.2,)}})
    decision = backend.decide("state", questions)
    assert isinstance(backend, Backend)
    assert decision.answers["kind"].confidence == 0.9
    assert decision.answers["dup"].confidence is None
    assert decision.answers["dup"].probabilities == {"true": 0.2, "false": 0.8}
    assert decision.usage.input_tokens == 100
