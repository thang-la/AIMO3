from __future__ import annotations

import argparse
import os

from aimo3.controller import AIMO3Solver


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("problem", type=str, help="Problem statement text")
    ap.add_argument("--pid", type=str, default="local")
    args = ap.parse_args()

    seed = int.from_bytes(os.urandom(8), "big")
    solver = AIMO3Solver()
    ans = solver.solve_one(args.pid, args.problem, seed)
    print(ans)


if __name__ == "__main__":
    main()
