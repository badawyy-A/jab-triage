import pytest

from jab_triage import backends
from jab_triage.backends.base import BackendError
from tests.stub import ScriptedBackend


def test_unset_backend_is_a_clear_error():
    with pytest.raises(BackendError, match="JEV_BACKEND is not set"):
        backends.from_env({})


def test_unknown_backend_is_a_clear_error():
    with pytest.raises(BackendError, match="JEV_BACKEND='nope' is not a known backend"):
        backends.from_env({"JEV_BACKEND": "nope"})


def test_known_backend_is_built_from_env(monkeypatch):
    stub = ScriptedBackend({})
    monkeypatch.setitem(backends.BACKENDS, "stub", lambda env: stub)
    assert backends.from_env({"JEV_BACKEND": " stub "}) is stub
