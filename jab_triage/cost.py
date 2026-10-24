"""Add up token usage and cost across a run.

Cost comes from the API when it reports one (OpenRouter does). When it does not
(TypeSafe direct never does), it is computed from the documented price, and the
summary says which it was.
"""

from __future__ import annotations

from dataclasses import dataclass

from jab_triage.backends.base import Usage

# USD per million input tokens for jev-1.13 and jev-1.14, on both TypeSafe and
# OpenRouter, as documented on 2026-10-24. Output tokens are free.
INPUT_PRICE_PER_MILLION = 0.042


@dataclass
class CostTally:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    reported_cost: float = 0.0
    calls_with_cost: int = 0
    # Input tokens of the calls that came back without a cost.
    uncosted_input_tokens: int = 0

    def add(self, usage: Usage) -> None:
        self.calls += 1
        self.input_tokens += usage.input_tokens
        self.output_tokens += usage.output_tokens
        if usage.cost is None:
            self.uncosted_input_tokens += usage.input_tokens
        else:
            self.reported_cost += usage.cost
            self.calls_with_cost += 1

    def to_json(self) -> dict:
        computed = self.uncosted_input_tokens * INPUT_PRICE_PER_MILLION / 1_000_000
        price = f"the documented price of ${INPUT_PRICE_PER_MILLION} per million input tokens"
        if self.calls_with_cost == self.calls:
            source = "reported by api"
        elif self.calls_with_cost == 0:
            source = f"computed from {price}"
        else:
            source = (
                f"reported by api for {self.calls_with_cost} of {self.calls} calls, "
                f"the rest computed from {price}"
            )
        return {
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            # Rounded to cut float noise; a call costs far more than 1e-10 USD.
            "cost_usd": round(self.reported_cost + computed, 10),
            "cost_source": source,
        }
