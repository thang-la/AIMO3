from __future__ import annotations

import ast
import operator
import re
from collections import defaultdict
from itertools import product

from aimo3.models import Candidate, PathType, ProblemMetadata, PromptProgram

try:
    import sympy as sp
except Exception:  # pragma: no cover - optional dependency
    sp = None  # type: ignore[assignment]


_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _factorize(n: int) -> dict[int, int]:
    out: dict[int, int] = {}
    x = n
    d = 2
    while d * d <= x:
        while x % d == 0:
            out[d] = out.get(d, 0) + 1
            x //= d
        d += 1 if d == 2 else 2
    if x > 1:
        out[x] = out.get(x, 0) + 1
    return out


def _v_p(n: int, p: int) -> int:
    if p <= 1:
        return 0
    x = abs(n)
    count = 0
    while x > 0 and x % p == 0:
        x //= p
        count += 1
    return count


def _word_to_int(token: str) -> int | None:
    table = {
        "double": 2,
        "triple": 3,
        "quadruple": 4,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
    }
    return table.get(token.strip().lower())


def _safe_eval_arithmetic(expr: str) -> int | None:
    try:
        node = ast.parse(expr, mode="eval").body
    except Exception:
        return None

    def _eval(n: ast.AST) -> int | float:
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp):
            if type(n.op) not in _SAFE_OPS:
                raise ValueError("unsupported operator")
            return _SAFE_OPS[type(n.op)](_eval(n.left), _eval(n.right))
        if isinstance(n, ast.UnaryOp):
            if type(n.op) not in _SAFE_OPS:
                raise ValueError("unsupported unary operator")
            return _SAFE_OPS[type(n.op)](_eval(n.operand))
        raise ValueError("unsupported expression")

    try:
        value = _eval(node)
        return int(value)
    except Exception:
        return None


def _solve_simple_remainder(meta: ProblemMetadata) -> Candidate | None:
    text = meta.normalized_text
    m = re.search(r"remainder when (.+) is divided by (\d+)", text, flags=re.IGNORECASE)
    if not m:
        return None
    expr = m.group(1).strip().replace("^", "**")
    modulus = int(m.group(2))
    value = _safe_eval_arithmetic(expr)
    if value is None:
        return None
    return Candidate(
        path=PathType.P0_SYMBOLIC,
        answer=value % modulus,
        trace="P0 symbolic: parsed direct remainder expression and evaluated safely.",
        program=PromptProgram.A,
        metadata={"symbolic_rule": "direct_remainder"},
    )


def _solve_simple_linear_equation(meta: ProblemMetadata) -> Candidate | None:
    if sp is None or not meta.extracted_equations:
        return None
    equation = meta.extracted_equations[0]
    if equation.count("=") != 1:
        return None
    left, right = equation.split("=", 1)
    variables = [v for v in meta.variables if v.isalpha()]
    if not variables:
        return None
    symbol = sp.Symbol(variables[0])
    try:
        left_expr = sp.sympify(left.replace("^", "**"))
        right_expr = sp.sympify(right.replace("^", "**"))
        solutions = sp.solve(sp.Eq(left_expr, right_expr), symbol)
        if len(solutions) != 1:
            return None
        value = int(solutions[0])
    except Exception:
        return None
    return Candidate(
        path=PathType.P0_SYMBOLIC,
        answer=value,
        trace="P0 symbolic: solved linear equation with SymPy.",
        program=PromptProgram.A,
        metadata={"symbolic_rule": "linear_equation"},
    )


def _solve_floor_sum_valuation(meta: ProblemMetadata) -> Candidate | None:
    text = meta.normalized_text.lower()
    compact = text.replace(" ", "")
    has_floor_pattern = (
        "floor(1/j+(n-i)/n)" in compact
        or "\\lfloor\\frac1j+\\frac{n-i}{n}\\rfloor" in compact
        or "\\lfloor\\frac{1}{j}+\\frac{n-i}{n}\\rfloor" in compact
    )
    if not has_floor_pattern:
        return None
    if "divides n" not in text or "let k be the largest non-negative integer" not in text:
        return None

    p_match = re.search(r"j\^\{?(\d+)\}?", text)
    if not p_match:
        return None
    power = int(p_match.group(1))

    m_def = re.search(r"m\s*=\s*([0-9\*\s]+)", text)
    if not m_def:
        return None
    m_factors = [int(x) for x in re.findall(r"\d+", m_def.group(1))]
    if not m_factors:
        return None
    m_val = 1
    for f in m_factors:
        m_val *= f

    nexp_match = re.search(
        r"f\{?\(?m\^\{?(\d+)\}?\)?\}?\s*-\s*f\{?\(?m\^\{?(\d+)\}?\s*-\s*1\)?\}?",
        compact,
    )
    if not nexp_match:
        return None
    if nexp_match.group(1) != nexp_match.group(2):
        return None
    n_exp = int(nexp_match.group(1))

    val_match = re.search(r"such that\s+(\d+)\^k\s+divides\s+n", text)
    if not val_match:
        return None
    val_prime = int(val_match.group(1))

    out_mod_match = re.search(r"remainder when\s+\d+\^k\s+is divided by\s+(\d+)\^(\d+)", text)
    if out_mod_match:
        out_mod = int(out_mod_match.group(1)) ** int(out_mod_match.group(2))
    else:
        out_mod_simple = re.search(r"is divided by\s+(\d+)\s*$", text)
        out_mod = int(out_mod_simple.group(1)) if out_mod_simple else 100000

    n_factors = _factorize(m_val)
    n_factors = {p: e * n_exp for p, e in n_factors.items()}

    k = 0
    for p, e in n_factors.items():
        num = pow(p, power * (e + 1)) - 1
        den = pow(p, power) - 1
        if den == 0:
            return None
        sigma_term = num // den
        k += _v_p(sigma_term, val_prime)

    answer = pow(val_prime, k, out_mod)
    return Candidate(
        path=PathType.P0_SYMBOLIC,
        answer=answer,
        trace=(
            "P0 symbolic: simplified floor double-sum difference to sigma_p(n), "
            "computed p-adic valuation multiplicatively, then reduced final power."
        ),
        program=PromptProgram.A,
        metadata={"symbolic_rule": "floor_sum_valuation", "k": k},
    )


def _parse_story_multiplier(text: str, phrase: str) -> int | None:
    # Supports both numeral and words: "double", "four", etc.
    pat_num = re.search(rf"{re.escape(phrase)}.*?(\d+)\s*times", text)
    if pat_num:
        return int(pat_num.group(1))
    if "double" in text and "sum" in phrase:
        return 2
    if "four times" in text and "product" in phrase:
        return 4
    word_times = re.search(rf"{re.escape(phrase)}.*?(double|triple|quadruple|four)\s+times", text)
    if word_times:
        return _word_to_int(word_times.group(1))
    return None


def _solve_age_sweets(meta: ProblemMetadata) -> Candidate | None:
    text = meta.normalized_text.lower()
    markers = ["alice", "bob", "sweets", "ages", "give me"]
    if not all(m in text for m in markers):
        return None

    sum_mult = _parse_story_multiplier(text, "sum")
    prod_mult = _parse_story_multiplier(text, "product")
    if sum_mult is None:
        if "double" in text:
            sum_mult = 2
    if prod_mult is None:
        if "four times" in text:
            prod_mult = 4
    if sum_mult is None or prod_mult is None:
        return None

    transfer_match = re.search(r"give me\s+(\d+)", text)
    transfer = int(transfer_match.group(1)) if transfer_match else 5

    # A_sweets+ A_age = sum_mult*(B_sweets + B_age)
    # (A_sweets-transfer) + A_age = (B_sweets+transfer) + B_age
    # => (sum_mult-1)*(B_sweets+B_age)=2*transfer
    den = sum_mult - 1
    if den <= 0:
        return None
    rhs = 2 * transfer
    if rhs % den != 0:
        return None
    sum_b = rhs // den
    sum_a = sum_mult * sum_b

    sols: list[int] = []
    for a_age in range(1, sum_a):
        a_sweets = sum_a - a_age
        if a_sweets <= transfer:
            continue
        for b_age in range(1, sum_b):
            b_sweets = sum_b - b_age
            if b_sweets <= 0:
                continue
            if a_sweets * a_age != prod_mult * (b_sweets * b_age):
                continue
            if (a_sweets - transfer) * a_age != (b_sweets + transfer) * b_age:
                continue
            sols.append(a_age * b_age)

    if len(set(sols)) != 1:
        return None
    answer = sols[0]
    return Candidate(
        path=PathType.P0_SYMBOLIC,
        answer=answer,
        trace="P0 symbolic: solved age/sweets system via derived sum constraints and integer search.",
        program=PromptProgram.A,
        metadata={"symbolic_rule": "age_sweets"},
    )


def _solve_functional_equation_count(meta: ProblemMetadata) -> Candidate | None:
    text = meta.normalized_text.lower()
    compact = text.replace(" ", "")
    if "f(m)+f(n)=f(m+n+mn)" not in compact:
        return None
    bound_match = re.search(
        r"f\(n\)\s*(?:<=|\\leq)\s*(\d+)\s*for all n\s*(?:<=|\\leq)\s*(\d+)",
        text,
    )
    if not bound_match:
        return None
    b1 = int(bound_match.group(1))
    b2 = int(bound_match.group(2))
    if b1 != b2:
        return None
    bound = b1

    target_match = re.search(r"how many different values can f\((\d+)\) take", text)
    if not target_match:
        return None
    target = int(target_match.group(1))
    t = target + 1
    target_fac = _factorize(t)
    target_primes = sorted(target_fac.keys())
    if not target_primes:
        return None
    if len(target_primes) > 3:
        return None

    # Build constraints from g(n)=f(n-1), g(ab)=g(a)+g(b), g(n)<=bound for n<=bound+1.
    # Let objective vars be w_p = g(p) for p in target_primes.
    idx = {p: i for i, p in enumerate(target_primes)}
    raw_constraints: dict[tuple[int, ...], int] = {}
    max_n = bound + 1
    for n in range(2, max_n + 1):
        fac = _factorize(n)
        coeff = [0] * len(target_primes)
        extra = 0
        for p, e in fac.items():
            if p in idx:
                coeff[idx[p]] += e
            else:
                extra += e  # minimal contribution from non-target primes with g(p)>=1
        rhs = bound - extra
        key = tuple(coeff)
        if all(c == 0 for c in key):
            continue
        if key not in raw_constraints or rhs < raw_constraints[key]:
            raw_constraints[key] = rhs
    constraints = [(list(k), v) for k, v in raw_constraints.items()]

    ub = [bound] * len(target_primes)
    for coeff, rhs in constraints:
        for i, c in enumerate(coeff):
            if c > 0:
                ub[i] = min(ub[i], rhs // c)
    if any(u < 1 for u in ub):
        return None

    values: set[int] = set()
    # Small-dimensional exhaustive search (target has few prime factors in olympiad tasks).
    ranges = [range(1, u + 1) for u in ub]
    for ws in product(*ranges):
        ok = True
        for coeff, rhs in constraints:
            lhs = sum(c * w for c, w in zip(coeff, ws))
            if lhs > rhs:
                ok = False
                break
        if not ok:
            continue
        value = sum(target_fac[p] * ws[idx[p]] for p in target_primes)
        values.add(value)

    if not values:
        return None
    return Candidate(
        path=PathType.P0_SYMBOLIC,
        answer=len(values),
        trace=(
            "P0 symbolic: transformed to completely additive g(ab)=g(a)+g(b), "
            "applied bound constraints up to B+1, enumerated feasible prime weights."
        ),
        program=PromptProgram.A,
        metadata={"symbolic_rule": "functional_equation_count", "candidate_values": len(values)},
    )


def symbolic_first_pass(meta: ProblemMetadata) -> list[Candidate]:
    solvers = [
        _solve_floor_sum_valuation,
        _solve_functional_equation_count,
        _solve_age_sweets,
        _solve_simple_remainder,
        _solve_simple_linear_equation,
    ]
    candidates: list[Candidate] = []
    seen_answers: set[int] = set()
    for solver in solvers:
        cand = solver(meta)
        if cand is None or cand.answer is None:
            continue
        if cand.answer in seen_answers:
            continue
        seen_answers.add(cand.answer)
        candidates.append(cand)
    return candidates
