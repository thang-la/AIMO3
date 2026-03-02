from __future__ import annotations

import ast
import contextlib
import io
import multiprocessing as mp
import time
from dataclasses import dataclass
from typing import Any

_ALLOWED_IMPORTS = {"math", "fractions", "itertools", "sympy", "numpy"}
_ALLOWED_NODES = {
    ast.Module,
    ast.Assign,
    ast.AnnAssign,
    ast.Expr,
    ast.Call,
    ast.Name,
    ast.Load,
    ast.Store,
    ast.Constant,
    ast.BinOp,
    ast.UnaryOp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.Compare,
    ast.Eq,
    ast.NotEq,
    ast.Gt,
    ast.GtE,
    ast.Lt,
    ast.LtE,
    ast.If,
    ast.For,
    ast.While,
    ast.Break,
    ast.Continue,
    ast.List,
    ast.Tuple,
    ast.Dict,
    ast.Set,
    ast.Subscript,
    ast.Slice,
    ast.ListComp,
    ast.DictComp,
    ast.SetComp,
    ast.comprehension,
    ast.IfExp,
    ast.BoolOp,
    ast.And,
    ast.Or,
    ast.Not,
    ast.USub,
    ast.UAdd,
    ast.FunctionDef,
    ast.arguments,
    ast.arg,
    ast.Return,
    ast.Pass,
    ast.Import,
    ast.ImportFrom,
    ast.Try,
    ast.ExceptHandler,
}
_BLOCKED_NAMES = {
    "open",
    "exec",
    "eval",
    "compile",
    "__import__",
    "input",
    "help",
    "globals",
    "locals",
    "vars",
    "dir",
    "getattr",
    "setattr",
    "delattr",
}


@dataclass(slots=True)
class SandboxResult:
    success: bool
    answer: int | None
    stdout: str
    stderr: str
    runtime_ms: float
    deterministic_score: float
    flags: list[str]


def _validate_ast(tree: ast.AST) -> list[str]:
    flags: list[str] = []
    for node in ast.walk(tree):
        if type(node) not in _ALLOWED_NODES:
            flags.append(f"node_blocked:{type(node).__name__}")
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in _ALLOWED_IMPORTS:
                    flags.append(f"import_blocked:{alias.name}")
        if isinstance(node, ast.ImportFrom):
            mod = (node.module or "").split(".")[0]
            if mod not in _ALLOWED_IMPORTS:
                flags.append(f"importfrom_blocked:{node.module}")
        if isinstance(node, ast.Name) and node.id in _BLOCKED_NAMES:
            flags.append(f"name_blocked:{node.id}")
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if "/" in node.value or "\\" in node.value:
                flags.append("path_literal_detected")
    return flags


def _run_code(code: str, queue: mp.Queue) -> None:
    start = time.perf_counter()
    out_buf = io.StringIO()
    err_buf = io.StringIO()
    answer = None
    success = False
    flags: list[str] = []
    with contextlib.redirect_stdout(out_buf), contextlib.redirect_stderr(err_buf):
        try:
            env: dict[str, Any] = {
                "__builtins__": {
                    "abs": abs,
                    "all": all,
                    "any": any,
                    "enumerate": enumerate,
                    "int": int,
                    "float": float,
                    "range": range,
                    "len": len,
                    "sum": sum,
                    "min": min,
                    "max": max,
                    "print": print,
                    "zip": zip,
                    "map": map,
                    "filter": filter,
                    "list": list,
                    "tuple": tuple,
                    "dict": dict,
                    "set": set,
                    "sorted": sorted,
                }
            }
            exec(code, env, env)
            if "ANSWER" in env:
                answer = int(env["ANSWER"])
            else:
                for line in out_buf.getvalue().splitlines()[::-1]:
                    if "ANSWER" in line.upper():
                        parts = [p for p in line.replace("=", " ").split() if p.lstrip("-").isdigit()]
                        if parts:
                            answer = int(parts[-1])
                            break
            success = answer is not None
        except Exception as exc:  # pragma: no cover - defensive
            err_buf.write(str(exc))
    runtime_ms = (time.perf_counter() - start) * 1000.0
    queue.put(
        {
            "success": success,
            "answer": answer,
            "stdout": out_buf.getvalue(),
            "stderr": err_buf.getvalue(),
            "runtime_ms": runtime_ms,
            "flags": flags,
        }
    )


def run_in_sandbox(code: str, timeout_s: float = 2.0, repeat: int = 3) -> SandboxResult:
    parse_flags: list[str] = []
    try:
        tree = ast.parse(code)
        parse_flags = _validate_ast(tree)
    except SyntaxError as exc:
        return SandboxResult(
            success=False,
            answer=None,
            stdout="",
            stderr=f"SyntaxError: {exc}",
            runtime_ms=0.0,
            deterministic_score=0.0,
            flags=["syntax_error"],
        )

    if parse_flags:
        return SandboxResult(
            success=False,
            answer=None,
            stdout="",
            stderr="blocked by AST policy",
            runtime_ms=0.0,
            deterministic_score=0.0,
            flags=parse_flags,
        )

    answers: list[int | None] = []
    last_payload: dict[str, Any] = {
        "success": False,
        "answer": None,
        "stdout": "",
        "stderr": "",
        "runtime_ms": 0.0,
        "flags": [],
    }

    for _ in range(max(1, repeat)):
        q: mp.Queue = mp.Queue(maxsize=1)
        p = mp.Process(target=_run_code, args=(code, q), daemon=True)
        p.start()
        p.join(timeout=timeout_s)
        if p.is_alive():
            p.terminate()
            p.join()
            return SandboxResult(
                success=False,
                answer=None,
                stdout="",
                stderr="timeout",
                runtime_ms=timeout_s * 1000.0,
                deterministic_score=0.0,
                flags=["timeout"],
            )
        if not q.empty():
            last_payload = q.get()
        answers.append(last_payload.get("answer"))

    non_none = [a for a in answers if a is not None]
    if not non_none:
        det = 0.0
    else:
        same = sum(1 for a in answers if a == non_none[0])
        det = same / len(answers)

    return SandboxResult(
        success=bool(last_payload.get("success")),
        answer=last_payload.get("answer"),
        stdout=str(last_payload.get("stdout", "")),
        stderr=str(last_payload.get("stderr", "")),
        runtime_ms=float(last_payload.get("runtime_ms", 0.0)),
        deterministic_score=det,
        flags=list(last_payload.get("flags", [])),
    )

