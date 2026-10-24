"""The interface every Jev backend implements.

Triage code only talks to this interface, so the transport (OpenRouter or
TypeSafe direct) can change without touching anything else.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from jab_triage.presets import Question


class BackendError(Exception):
    """A call failed. The message must never contain the API key or item text."""


@dataclass(frozen=True)
class Answer:
    type: str
    # The option name for choice, the expected level for score, the probability of yes for noul.
    value: Any
    # For noul this is {"true": p, "false": 1 - p}, so every type has a distribution.
    probabilities: dict[str, float]
    # The API's own confidence. None when it does not return one, which is the case for noul.
    confidence: float | None


@dataclass(frozen=True)
class Usage:
    input_tokens: int
    output_tokens: int
    # USD, only when the API reports it.
    cost: float | None = None


@dataclass(frozen=True)
class Decision:
    answers: dict[str, Answer]
    usage: Usage
    # The model that answered, as the response names it.
    model: str


class Backend(ABC):
    name: str

    @abstractmethod
    def decide(self, state: Any, questions: tuple[Question, ...]) -> Decision:
        """Ask all questions about one item's state in a single call.

        Raise BackendError when the call fails.
        """
