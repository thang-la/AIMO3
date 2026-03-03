import csv
from pathlib import Path

from aimo3.parsing import parse_problem
from aimo3.symbolic import symbolic_first_pass


def _problem_by_id(pid: str) -> str:
    path = Path("reference.csv")
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["id"] == pid:
                return row["problem"]
    raise AssertionError(f"Missing problem id: {pid}")


def _has_answer(cands, answer: int) -> bool:
    return any(c.answer == answer for c in cands)


def test_floor_sum_valuation_pattern():
    text = _problem_by_id("26de63")
    meta = parse_problem("26de63", text)
    cands = symbolic_first_pass(meta)
    assert _has_answer(cands, 32951)


def test_age_sweets_pattern():
    text = _problem_by_id("92ba6a")
    meta = parse_problem("92ba6a", text)
    cands = symbolic_first_pass(meta)
    assert _has_answer(cands, 50)


def test_functional_equation_pattern():
    text = _problem_by_id("9c1c5f")
    meta = parse_problem("9c1c5f", text)
    cands = symbolic_first_pass(meta)
    assert _has_answer(cands, 580)
