"""Pick the backend named by JEV_BACKEND."""

from __future__ import annotations

from collections.abc import Callable, Mapping

from jab_triage.backends.base import Backend, BackendError

# JEV_BACKEND value to a function that builds that backend from the environment.
BACKENDS: dict[str, Callable[[Mapping[str, str]], Backend]] = {}

# The environment variable each backend reads its key from.
KEY_VARS = {"openrouter": "OPENROUTER_API_KEY", "typesafe": "TYPESAFE_API_KEY"}


def from_env(env: Mapping[str, str]) -> Backend:
    name = env.get("JEV_BACKEND", "").strip()
    choices = ", ".join(sorted(BACKENDS)) or "none available yet"
    if not name:
        raise BackendError(f"JEV_BACKEND is not set. Choices: {choices}")
    factory = BACKENDS.get(name)
    if factory is None:
        raise BackendError(f"JEV_BACKEND={name!r} is not a known backend. Choices: {choices}")
    return factory(env)
