import threading
import time

import pytest

from jab_triage.backends.base import FatalBackendError
from jab_triage.triage import run
from tests.stub import ScriptedBackend
from tests.test_triage import PRESET, items

SURE = {"kind": ("bug", 0.9), "dup": (0.9,)}


def test_calls_run_in_parallel():
    barrier = threading.Barrier(4, timeout=5)

    class Waiting(ScriptedBackend):
        def decide(self, state, questions):
            # Only gets past this when four calls are in flight at once.
            barrier.wait()
            return super().decide(state, questions)

    texts = [f"t{i}" for i in range(4)]
    result = run(Waiting(dict.fromkeys(texts, SURE)), PRESET, items(*texts), concurrency=4)
    assert [r.error for r in result.items] == [None] * 4


def test_results_keep_input_order():
    class Delayed(ScriptedBackend):
        def decide(self, state, questions):
            # Later items finish first.
            time.sleep(0.01 * (5 - int(state[1:])))
            return super().decide(state, questions)

    texts = [f"t{i}" for i in range(5)]
    result = run(Delayed(dict.fromkeys(texts, SURE)), PRESET, items(*texts), concurrency=5)
    assert [r.id for r in result.items] == [0, 1, 2, 3, 4]


def test_fatal_error_stops_sending_new_items():
    class Slow(ScriptedBackend):
        def decide(self, state, questions):
            if state != "t0":
                time.sleep(0.05)
            return super().decide(state, questions)

    texts = [f"t{i}" for i in range(20)]
    script = dict.fromkeys(texts, SURE)
    script["t0"] = FatalBackendError("HTTP 401 from stub")
    backend = Slow(script)
    with pytest.raises(FatalBackendError):
        run(backend, PRESET, items(*texts), concurrency=2)
    # t0 failed at once; only t1, already in flight, went out after it.
    assert len(backend.calls) <= 2


def test_summary_records_concurrency_and_time():
    result = run(ScriptedBackend({"a": SURE}), PRESET, items("a"), concurrency=3)
    summary = result.summary()
    assert summary["concurrency"] == 3
    assert summary["seconds"] >= 0
