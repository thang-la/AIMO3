from __future__ import annotations

import ast
import operator
import re

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
    expr = m.group(1).strip()
    modulus = int(m.group(2))
    expr = expr.replace("^", "**")
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
    if sp is None:
        return None
    if not meta.extracted_equations:
        return None
    # This conservative rule handles patterns like "x + 5 = 17".
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


def symbolic_first_pass(meta: ProblemMetadata) -> list[Candidate]:
    candidates: list[Candidate] = []
    direct = _solve_simple_remainder(meta)
    if direct is not None:
        candidates.append(direct)
    linear = _solve_simple_linear_equation(meta)
    if linear is not None:
        candidates.append(linear)
    return candidates
