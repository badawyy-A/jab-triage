"""Add up token usage and cost across a run, from what the API reported."""

from __future__ import annotations

from dataclasses import dataclass

from jab_triage.backends.base import Usage


@dataclass
class CostTally:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    reported_cost: float = 0.0
    calls_with_cost: int = 0

    def add(self, usage: Usage) -> None:
        self.calls += 1
        self.input_tokens += usage.input_tokens
        self.output_tokens += usage.output_tokens
        if usage.cost is not None:
            self.reported_cost += usage.cost
            self.calls_with_cost += 1

    def to_json(self) -> dict:
        # A total is only given when every call reported its cost. A partial sum
        # would look like a real number while undercounting.
        if self.calls_with_cost == self.calls:
            cost, source = self.reported_cost, "reported by api"
        elif self.calls_with_cost == 0:
            cost, source = None, "not reported by api"
        else:
            cost = None
            source = f"reported for {self.calls_with_cost} of {self.calls} calls"
        return {
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": cost,
            "cost_source": source,
        }
