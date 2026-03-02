from __future__ import annotations

import os

from aimo3.config import SolverConfig
from aimo3.controller import AIMO3Solver
from aimo3.runtime import solver_config_from_env

_SOLVER: AIMO3Solver | None = None
_RUN_SEED = int.from_bytes(os.urandom(4), "big")


def get_solver() -> AIMO3Solver:
    global _SOLVER
    if _SOLVER is None:
        config: SolverConfig = solver_config_from_env()
        _SOLVER = AIMO3Solver(config=config)
    return _SOLVER


def _series_item(series) -> str:
    if hasattr(series, "item"):
        return series.item(0)
    if isinstance(series, list):
        return series[0]
    return str(series)


def predict(id_series, problem_series):
    solver = get_solver()
    pid = str(_series_item(id_series))
    problem = str(_series_item(problem_series))
    result = solver.solve_one(pid, problem, run_seed=_RUN_SEED)

    try:
        import polars as pl

        return pl.DataFrame({"id": [pid], "answer": [int(result.answer)]})
    except Exception:
        try:
            import pandas as pd

            return pd.DataFrame({"id": [pid], "answer": [int(result.answer)]})
        except Exception:
            return {"id": [pid], "answer": [int(result.answer)]}
