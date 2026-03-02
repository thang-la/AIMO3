from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aimo3.controller import AIMO3Solver
from aimo3.models import DeterministicHeuristicModel


def main() -> None:
    stub = DeterministicHeuristicModel()
    solver = AIMO3Solver(main_model=stub, fast_model=stub)
    ans = solver.solve_one("smoke", "Find the remainder when 123456 is divided by 97.", run_seed=123)
    print({"answer": ans})


if __name__ == "__main__":
    main()
