from dataclasses import replace
from pathlib import Path

from aimo3.config import SolverConfig
from aimo3.controller import AIMO3Solver
from aimo3.kaggle_server import predict


def test_solver_smoke():
    cfg = SolverConfig()
    cfg = replace(cfg, llm=replace(cfg.llm, backend="heuristic"), allow_demo_fallback=True, enforce_real_backend=False)
    solver = AIMO3Solver(config=cfg)
    result = solver.solve_one("t0", "What is the remainder when 123456 is divided by 97?")
    assert isinstance(result.answer, int)
    assert 0 <= result.answer <= 99999


def test_kaggle_predict_smoke():
    import os

    os.environ["AIMO3_BACKEND"] = "heuristic"
    os.environ["AIMO3_ALLOW_DEMO_FALLBACK"] = "1"
    os.environ["AIMO3_ENFORCE_REAL_BACKEND"] = "0"
    out = predict(["x1"], ["What is the remainder when 11^5 is divided by 13?"])
    if hasattr(out, "to_dict"):
        payload = out.to_dict(as_series=False) if "polars" in str(type(out)).lower() else out.to_dict()
        assert "id" in payload
        assert "answer" in payload
    else:
        assert "id" in out
        assert "answer" in out


def test_memory_lookup_exact(tmp_path: Path):
    reference = tmp_path / "reference.csv"
    reference.write_text(
        '"id","problem","answer"\n'
        '"m1","What is the remainder when 123456 is divided by 97?","72"\n',
        encoding="utf-8",
    )
    cfg = SolverConfig()
    cfg = replace(
        cfg,
        llm=replace(cfg.llm, backend="heuristic"),
        allow_demo_fallback=True,
        enforce_real_backend=False,
        allow_reference_lookup=True,
        reference_path=reference,
    )
    solver = AIMO3Solver(config=cfg)
    result = solver.solve_one("m1", "What is the remainder when 123456 is divided by 97?")
    assert result.answer == 72
    assert "memory_match" in result.reason
