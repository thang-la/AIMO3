from __future__ import annotations

import os

try:  # pragma: no cover - runtime dependency may be unavailable in local env
    import polars as pl
except Exception:  # pragma: no cover
    class _PLShim:
        @staticmethod
        def DataFrame(data: dict):
            return data

    pl = _PLShim()

from .controller import AIMO3Solver


_SOLVER: AIMO3Solver | None = None
_RUN_SEED: int = int.from_bytes(os.urandom(8), "big")


def _get_solver() -> AIMO3Solver:
    global _SOLVER
    if _SOLVER is None:
        _SOLVER = AIMO3Solver()
    return _SOLVER


def predict(id_series, problem_series) -> pl.DataFrame:
    pid = str(id_series.item(0))
    problem = str(problem_series.item(0))
    ans = _get_solver().solve_one(pid, problem, _RUN_SEED)
    return pl.DataFrame({"id": [pid], "answer": [int(ans)]})
