import pytest

from jab_triage.backends.base import Usage
from jab_triage.cost import INPUT_PRICE_PER_MILLION, CostTally


def test_sums_tokens_and_reported_cost():
    tally = CostTally()
    tally.add(Usage(100, 10, 0.001))
    tally.add(Usage(50, 5, 0.002))
    summary = tally.to_json()
    assert summary["calls"] == 2
    assert summary["input_tokens"] == 150
    assert summary["output_tokens"] == 15
    assert summary["cost_usd"] == pytest.approx(0.003)
    assert summary["cost_source"] == "reported by api"


def test_no_calls_costs_nothing():
    summary = CostTally().to_json()
    assert summary["calls"] == 0
    assert summary["cost_usd"] == 0.0


def test_cost_computed_from_price_when_not_reported():
    tally = CostTally()
    tally.add(Usage(1_000_000, 10))
    summary = tally.to_json()
    assert summary["cost_usd"] == pytest.approx(INPUT_PRICE_PER_MILLION)
    assert summary["cost_source"].startswith("computed from the documented price")


def test_partial_cost_fills_the_gap_from_price():
    tally = CostTally()
    tally.add(Usage(100, 10, 0.001))
    tally.add(Usage(1_000_000, 10))
    summary = tally.to_json()
    assert summary["cost_usd"] == pytest.approx(0.001 + INPUT_PRICE_PER_MILLION)
    assert summary["cost_source"].startswith("reported by api for 1 of 2 calls")
