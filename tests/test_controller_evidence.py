from __future__ import annotations

from aimo3.belief import integrate_refutation_online
from aimo3.controller import AIMO3Solver
from aimo3.models import DeterministicHeuristicModel
from aimo3.types import RefuteEvent


def test_recompute_preserves_non_support_evidence() -> None:
    stub = DeterministicHeuristicModel()
    solver_a = AIMO3Solver(main_model=stub, fast_model=stub)
    solver_b = AIMO3Solver(main_model=stub, fast_model=stub)

    st_refute = solver_a.init_state("Find the remainder when 123456 is divided by 97.", run_seed=17)
    st_plain = solver_b.init_state("Find the remainder when 123456 is divided by 97.", run_seed=17)

    solver_a.execute_action_and_update(st_refute, {"type": "GEN", "policy": "A_TOOLFORMAL", "n": 1}, 17)
    solver_b.execute_action_and_update(st_plain, {"type": "GEN", "policy": "A_TOOLFORMAL", "n": 1}, 17)
    a = st_refute.candidates[0].answer

    integrate_refutation_online(st_refute, RefuteEvent(target_answer=a, outcome="FOUND_CONTRADICTION"), solver_a.cfg)

    # Trigger support recompute path again on both branches.
    solver_a.execute_action_and_update(st_refute, {"type": "GEN", "policy": "A_SYMBOLIC", "n": 1}, 17)
    solver_b.execute_action_and_update(st_plain, {"type": "GEN", "policy": "A_SYMBOLIC", "n": 1}, 17)

    # Branch with refutation should still keep lower score for the refuted answer.
    assert st_refute.s.get(a, 0.0) < st_plain.s.get(a, 0.0)
