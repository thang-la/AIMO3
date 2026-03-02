from __future__ import annotations

import ast
import io
import multiprocessing as mp
import re
import signal
import traceback
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass

_SAFE_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "pow": pow,
    "print": print,
    "range": range,
    "reversed": reversed,
    "round": round,
    "set": set,
    "sorted": sorted,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
}

_ALLOWED_IMPORTS = {"math", "fractions", "itertools", "sympy", "numpy", "collections"}
_ALLOWED_AST = {
    ast.Module,
    ast.Assign,
    ast.AugAssign,
    ast.Expr,
    ast.Load,
    ast.Store,
    ast.Name,
    ast.Constant,
    ast.BinOp,
    ast.UnaryOp,
    ast.BoolOp,
    ast.Compare,
    ast.Subscript,
    ast.Slice,
    ast.Tuple,
    ast.List,
    ast.Dict,
    ast.Set,
    ast.For,
    ast.If,
    ast.While,
    ast.Break,
    ast.Continue,
    ast.Pass,
    ast.Call,
    ast.keyword,
    ast.Return,
    ast.FunctionDef,
    ast.arguments,
    ast.arg,
    ast.Lambda,
    ast.Import,
    ast.ImportFrom,
    ast.Try,
    ast.ExceptHandler,
    ast.With,
    ast.comprehension,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
    ast.IfExp,
    ast.Attribute,
    ast.Mod,
    ast.Pow,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Add,
    ast.Sub,
    ast.USub,
    ast.UAdd,
    ast.And,
    ast.Or,
    ast.Not,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
}


@dataclass
class SandboxResult:
    success: bool
    answer: int | None
    stdout: str = ""
    stderr: str = ""
    timeout: bool = False
    error: str | None = None


def _validate_ast(code: str) -> None:
    tree = ast.parse(code, mode="exec")
    for node in ast.walk(tree):
        if type(node) not in _ALLOWED_AST:
            raise ValueError(f"Disallowed AST node: {type(node).__name__}")
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in _ALLOWED_IMPORTS:
                    raise ValueError(f"Disallowed import: {alias.name}")
        if isinstance(node, ast.ImportFrom):
            module = (node.module or "").split(".")[0]
            if module not in _ALLOWED_IMPORTS:
                raise ValueError(f"Disallowed import-from module: {node.module}")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise ValueError("Dunder names are disallowed")


def _extract_int_from_stdout(stdout: str) -> int | None:
    matches = re.findall(r"-?\d+", stdout)
    if not matches:
        return None
    return int(matches[-1])


def _worker(code: str, queue: mp.Queue, memory_mb: int) -> None:
    queue.put(_execute_code_capture(code, memory_mb))


def _execute_code_capture(code: str, memory_mb: int) -> dict[str, object]:
    try:
        try:
            import resource

            memory_bytes = int(memory_mb * 1024 * 1024)
            resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
        except Exception:
            # Resource limits are not available on all platforms.
            pass

        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        safe_globals = {"__builtins__": _SAFE_BUILTINS}
        safe_locals: dict[str, object] = {}

        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            exec(compile(code, "<sandbox>", "exec"), safe_globals, safe_locals)

        answer = None
        if "ANSWER" in safe_locals:
            answer = int(safe_locals["ANSWER"])  # type: ignore[arg-type]
        else:
            answer = _extract_int_from_stdout(stdout_buffer.getvalue())

        return {
            "success": answer is not None,
            "answer": answer,
            "stdout": stdout_buffer.getvalue(),
            "stderr": stderr_buffer.getvalue(),
        }
    except Exception as exc:
        return {
            "success": False,
            "answer": None,
            "stdout": "",
            "stderr": traceback.format_exc(),
            "error": str(exc),
        }


def _run_inline_with_timeout(code: str, timeout_s: float, memory_mb: int) -> SandboxResult:
    timed_out = False

    def _handle_timeout(_signum, _frame):  # pragma: no cover - signal dependent
        raise TimeoutError("timeout")

    old_handler = None
    try:
        if timeout_s > 0:
            old_handler = signal.getsignal(signal.SIGALRM)
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.setitimer(signal.ITIMER_REAL, timeout_s)
        payload = _execute_code_capture(code, memory_mb)
    except TimeoutError:
        timed_out = True
        payload = {"success": False, "answer": None, "stdout": "", "stderr": "", "error": "timeout"}
    finally:
        try:
            signal.setitimer(signal.ITIMER_REAL, 0)
        except Exception:
            pass
        if old_handler is not None:
            try:
                signal.signal(signal.SIGALRM, old_handler)
            except Exception:
                pass

    return SandboxResult(
        success=bool(payload.get("success")),
        answer=payload.get("answer"),  # type: ignore[arg-type]
        stdout=str(payload.get("stdout", "")),
        stderr=str(payload.get("stderr", "")),
        timeout=timed_out,
        error=payload.get("error"),  # type: ignore[arg-type]
    )


def run_python_sandbox(code: str, timeout_s: float, memory_mb: int) -> SandboxResult:
    try:
        _validate_ast(code)
    except Exception as exc:
        return SandboxResult(success=False, answer=None, error=str(exc))

    try:
        queue: mp.Queue = mp.Queue()
        process = mp.Process(target=_worker, args=(code, queue, memory_mb))
        process.start()
        process.join(timeout=timeout_s)

        if process.is_alive():
            process.terminate()
            process.join()
            return SandboxResult(success=False, answer=None, timeout=True, error="timeout")

        if queue.empty():
            return SandboxResult(success=False, answer=None, error="sandbox returned no result")

        payload = queue.get()
        return SandboxResult(
            success=bool(payload.get("success")),
            answer=payload.get("answer"),
            stdout=payload.get("stdout", ""),
            stderr=payload.get("stderr", ""),
            timeout=False,
            error=payload.get("error"),
        )
    except (PermissionError, OSError):
        # Some restricted runtimes disallow multiprocessing semaphores.
        return _run_inline_with_timeout(code, timeout_s=timeout_s, memory_mb=memory_mb)
