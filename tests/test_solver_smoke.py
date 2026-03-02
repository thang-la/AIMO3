from __future__ import annotations

from aimo3.controller import AIMO3Solver
from aimo3.models import DeterministicHeuristicModel


def test_solver_returns_int_in_range() -> None:
    stub = DeterministicHeuristicModel()
    solver = AIMO3Solver(main_model=stub, fast_model=stub)
    problem = "Find the remainder when 123456789 is divided by 97."
    ans = solver.solve_one("p1", problem, run_seed=42)
    assert isinstance(ans, int)
    assert 0 <= ans <= 99999
