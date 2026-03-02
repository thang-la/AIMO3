from aimo3.controller import AIMO3Solver
from aimo3.kaggle_server import predict


def test_solver_smoke():
    solver = AIMO3Solver()
    result = solver.solve_one("t0", "What is the remainder when 123456 is divided by 97?")
    assert isinstance(result.answer, int)
    assert 0 <= result.answer <= 99999


def test_kaggle_predict_smoke():
    out = predict(["x1"], ["What is the remainder when 11^5 is divided by 13?"])
    if hasattr(out, "to_dict"):
        payload = out.to_dict(as_series=False) if "polars" in str(type(out)).lower() else out.to_dict()
        assert "id" in payload
        assert "answer" in payload
    else:
        assert "id" in out
        assert "answer" in out
