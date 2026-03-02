from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from aimo3.controller import AIMO3Solver


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def cmd_solve_one(args: argparse.Namespace) -> int:
    solver = AIMO3Solver()
    problem = args.problem
    if args.problem_file:
        problem = Path(args.problem_file).read_text(encoding="utf-8")
    if not problem:
        raise SystemExit("Missing --problem or --problem-file")

    result = solver.solve_one(args.id, problem)
    if args.json:
        print(json.dumps(result.to_log_dict(), ensure_ascii=False, indent=2))
    else:
        print(result.answer)
    return 0


def cmd_solve_csv(args: argparse.Namespace) -> int:
    solver = AIMO3Solver()
    rows = _read_csv(Path(args.input))
    output_rows: list[dict[str, object]] = []
    score_correct = 0
    score_total = 0

    for row in rows:
        pid = str(row.get(args.id_col, "")).strip()
        problem = str(row.get(args.problem_col, ""))
        result = solver.solve_one(pid, problem)
        output_rows.append({"id": pid, "answer": int(result.answer)})
        if args.evaluate and "answer" in row and str(row["answer"]).strip():
            score_total += 1
            if int(row["answer"]) == int(result.answer):
                score_correct += 1

    _write_csv(Path(args.output), output_rows, fieldnames=["id", "answer"])
    print(f"Wrote {len(output_rows)} rows to {args.output}")
    if args.evaluate and score_total > 0:
        acc = score_correct / score_total
        print(f"Accuracy on provided answer column: {score_correct}/{score_total} = {acc:.2%}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AIMO3 end-to-end solver")
    sub = parser.add_subparsers(dest="command", required=True)

    p_one = sub.add_parser("solve-one", help="solve one problem")
    p_one.add_argument("--id", default="local-0")
    p_one.add_argument("--problem", default="")
    p_one.add_argument("--problem-file")
    p_one.add_argument("--json", action="store_true")
    p_one.set_defaults(func=cmd_solve_one)

    p_csv = sub.add_parser("solve-csv", help="solve CSV input -> submission")
    p_csv.add_argument("--input", required=True)
    p_csv.add_argument("--output", required=True)
    p_csv.add_argument("--id-col", default="id")
    p_csv.add_argument("--problem-col", default="problem")
    p_csv.add_argument("--evaluate", action="store_true")
    p_csv.set_defaults(func=cmd_solve_csv)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
