import pytest

from jab_triage.backends.base import Usage
from jab_triage.cost import CostTally


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


def test_cost_missing_from_every_call():
    tally = CostTally()
    tally.add(Usage(100, 10))
    summary = tally.to_json()
    assert summary["cost_usd"] is None
    assert summary["cost_source"] == "not reported by api"


def test_partial_cost_gives_no_total():
    tally = CostTally()
    tally.add(Usage(100, 10, 0.001))
    tally.add(Usage(100, 10))
    summary = tally.to_json()
    assert summary["cost_usd"] is None
    assert summary["cost_source"] == "reported for 1 of 2 calls"
