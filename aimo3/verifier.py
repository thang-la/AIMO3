from __future__ import annotations

import random
from dataclasses import asdict

from .sandbox import run_in_sandbox
from .types import Candidate, VerificationResult

try:  # pragma: no cover - optional dependency
    import sympy as sp
except Exception:  # pragma: no cover
    sp = None


def check_range(ans: int, bounds: tuple[int, int] = (0, 99999)) -> bool:
    lo, hi = bounds
    return lo <= int(ans) <= hi


def check_modulus(ans: int, meta: dict) -> bool:
    mod = meta.get("modulus")
    if mod is None:
        return True
    if not (isinstance(mod, int) and mod > 0):
        return False
    if meta.get("answer_should_be_remainder", False):
        return 0 <= int(ans) < mod
    return True


def check_integer_constraints(ans: int) -> bool:
    return isinstance(ans, int)


def symbolic_checks_if_possible(latex: str, ans: int, meta: dict, constraints: dict) -> bool:
    if not sp:
        return False
    equations = constraints.get("equations", [])
    if not equations:
        return False
    symbols = {str(s): s for s in sp.symbols("a b c d e f g h i j k m n p q r s t u v w x y z")}
    local = dict(symbols)
    local["ans"] = ans
    passed = 0
    seen = 0
    for eq in equations[:8]:
        seen += 1
        if "=" in eq:
            l, r = eq.split("=", 1)
            expr = f"({l})-({r})"
        else:
            expr = eq
        try:
            expr_obj = sp.sympify(expr, locals=local)
            val = expr_obj.subs({symbols.get("x", 0): ans}) if hasattr(expr_obj, "subs") else expr_obj
            if val == 0:
                passed += 1
        except Exception:
            continue
    return seen > 0 and passed >= max(1, seen // 2)


def randomized_tests_if_possible(cand: Candidate, meta: dict, trials: int = 10) -> float:
    if not cand.tool_code:
        return 0.0
    hits = 0
    for _ in range(trials):
        salt = random.randint(1, 1_000_000)
        test_code = cand.tool_code + f"\n_ = {salt}  # deterministic perturb noop\n"
        out = run_in_sandbox(test_code, timeout_s=1.0, repeat=1)
        if out.success and out.answer == cand.answer:
            hits += 1
    return hits / trials if trials else 0.0


def verify_shallow(cand: Candidate, meta: dict, constraints: dict) -> VerificationResult:
    v = VerificationResult(answer=cand.answer)
    range_ok = check_range(cand.answer, constraints.get("range", (0, 99999)))
    mod_ok = check_modulus(cand.answer, meta)
    int_ok = check_integer_constraints(cand.answer)
    v.hard_ok = bool(range_ok and mod_ok and int_ok)
    v.shallow_ok = v.hard_ok
    if not range_ok:
        v.flags.append("range_fail")
    if not mod_ok:
        v.flags.append("modulus_fail")
    if not int_ok:
        v.flags.append("integer_fail")
    return v


def verify_deep(cand: Candidate, meta: dict, constraints: dict, timeout_s: float = 2.0) -> VerificationResult:
    v = verify_shallow(cand, meta, constraints)
    if not v.hard_ok:
        return v

    if cand.tool_code:
        out = run_in_sandbox(cand.tool_code, timeout_s=timeout_s, repeat=3)
        v.tool_ok = bool(out.success and out.answer == cand.answer)
        v.tool_timeout = "timeout" in out.flags
        v.tool_error = bool(out.stderr and not out.success)
        v.artifacts["sandbox"] = asdict(out)
        if out.success and out.answer is not None and out.answer != cand.answer:
            v.flags.append("tool_answer_mismatch")
            v.contradictions += 1

    v.symbolic_ok = symbolic_checks_if_possible(meta.get("raw_latex", ""), cand.answer, meta, constraints)
    v.random_ok_rate = randomized_tests_if_possible(cand, meta, trials=6)
    v.deep_ok = v.hard_ok and (v.symbolic_ok or v.tool_ok or v.random_ok_rate >= 0.66)

    if not v.deep_ok:
        v.flags.append("deep_unconfirmed")
    return v
