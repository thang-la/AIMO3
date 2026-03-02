from __future__ import annotations

import json
from pathlib import Path

try:  # pragma: no cover
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LogisticRegression
except Exception:  # pragma: no cover
    np = None
    RandomForestRegressor = None
    LogisticRegression = None


def _vectorize(rows: list[dict]) -> tuple[list[str], list[list[float]], list[float], list[float], list[int]]:
    keys = sorted({k for row in rows for k in row.get("state_features", {}).keys()})
    x: list[list[float]] = []
    y_delta: list[float] = []
    y_cost: list[float] = []
    y_timeout: list[int] = []
    for row in rows:
        sf = row.get("state_features", {})
        x.append([float(sf.get(k, 0.0)) for k in keys])
        y_delta.append(float(row.get("delta_v", 0.0)))
        y_cost.append(float(row.get("cost", 0.0)))
        y_timeout.append(int(row.get("timeout", 0)))
    return keys, x, y_delta, y_cost, y_timeout


def train_avm(records_path: str | Path, out_path: str | Path) -> dict:
    rows = [json.loads(line) for line in Path(records_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    keys, x, y_delta, y_cost, y_timeout = _vectorize(rows)

    if np is None or RandomForestRegressor is None or LogisticRegression is None:
        payload = {"model": "heuristic", "feature_keys": keys}
        Path(out_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    x_np = np.asarray(x, dtype=float)
    delta_reg = RandomForestRegressor(n_estimators=200, random_state=0).fit(x_np, np.asarray(y_delta, dtype=float))
    cost_reg = RandomForestRegressor(n_estimators=200, random_state=1).fit(x_np, np.asarray(y_cost, dtype=float))
    timeout_clf = LogisticRegression(max_iter=1000).fit(x_np, np.asarray(y_timeout, dtype=int))

    payload = {
        "model": "rf+logreg",
        "feature_keys": keys,
        "delta_importances": [float(v) for v in delta_reg.feature_importances_],
        "cost_importances": [float(v) for v in cost_reg.feature_importances_],
        "timeout_coef": [float(v) for v in timeout_clf.coef_[0]],
        "timeout_bias": float(timeout_clf.intercept_[0]),
    }
    Path(out_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


if __name__ == "__main__":  # pragma: no cover
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("records")
    ap.add_argument("out")
    args = ap.parse_args()
    train_avm(args.records, args.out)

