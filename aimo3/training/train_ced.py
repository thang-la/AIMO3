from __future__ import annotations

import json
from pathlib import Path

try:  # pragma: no cover
    import numpy as np
    from sklearn.linear_model import HuberRegressor
except Exception:  # pragma: no cover
    np = None
    HuberRegressor = None


def _vectorize(rows: list[dict]) -> tuple[list[str], list[list[float]], list[float]]:
    keys = sorted({k for row in rows for k in row.get("pair_features", {}).keys()})
    x: list[list[float]] = []
    y: list[float] = []
    for row in rows:
        feats = row.get("pair_features", {})
        p_i = float(row.get("p_i", 0.5))
        p_j = float(row.get("p_j", 0.5))
        y_i = int(row.get("y_i", 0))
        y_j = int(row.get("y_j", 0))
        t = float(int(y_i == 0 and y_j == 0) - ((1.0 - p_i) * (1.0 - p_j)))
        x.append([float(feats.get(k, 0.0)) for k in keys])
        y.append(t)
    return keys, x, y


def train_ced(records_path: str | Path, out_path: str | Path) -> dict:
    rows = [json.loads(line) for line in Path(records_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    keys, x, y = _vectorize(rows)

    if HuberRegressor is None or np is None:
        payload = {"model": "heuristic", "feature_keys": keys, "weights": [0.0 for _ in keys], "bias": 0.0}
        Path(out_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    x_np = np.asarray(x, dtype=float)
    y_np = np.asarray(y, dtype=float)
    reg = HuberRegressor().fit(x_np, y_np)

    payload = {
        "model": "huber",
        "feature_keys": keys,
        "weights": [float(v) for v in reg.coef_],
        "bias": float(reg.intercept_),
    }
    Path(out_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


if __name__ == "__main__":  # pragma: no cover
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("records")
    ap.add_argument("out")
    args = ap.parse_args()
    train_ced(args.records, args.out)

