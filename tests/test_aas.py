from __future__ import annotations

import random

from aimo3.aas import select_answer, should_sample
from aimo3.config import DEFAULT_CONFIG
from aimo3.types import BeliefState, Budget


def make_state() -> BeliefState:
    return BeliefState(
        s={10: -0.6, 20: -0.7, "OTHER": -3.0},
        pi={10: 0.44, 20: 0.40, "OTHER": 0.16},
        u={10: 0.4, 20: 0.35, "OTHER": 1.0},
        clusters={},
        candidates=[],
        meta={"difficulty": 0.8},
        constraints={},
        run_config={"stop_pi": 0.92, "require_clusters": 2},
        budget=Budget(total_seconds=10, token_budget=1000, sandbox_runs_left=1),
        diagnostics={
            "cluster_support": {10: 1, 20: 2},
            "perturb_flip_rate": 0.35,
        },
        action_history=[],
    )


def test_should_sample_when_flat_and_fragile() -> None:
    st = make_state()
    assert should_sample(st, DEFAULT_CONFIG)


def test_select_answer_returns_valid_int() -> None:
    st = make_state()
    ans, meta = select_answer(st, DEFAULT_CONFIG, random.Random(0))
    assert ans in {10, 20}
    assert meta["mode"] in {"stochastic_ambiguous", "deterministic_default", "deterministic_high_conf"}

