from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:  # pragma: no cover - optional dependency
    import numpy as np
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
except Exception:  # pragma: no cover
    np = None
    IsotonicRegression = None
    LogisticRegression = None


def _vectorize(rows: list[dict]) -> tuple[list[str], list[list[float]], list[int]]:
    keys = sorted({k for row in rows for k in row.get("features", {}).keys()})
    x: list[list[float]] = []
    y: list[int] = []
    for row in rows:
        feats = row.get("features", {})
        x.append([float(feats.get(k, 0.0)) for k in keys])
        y.append(int(row.get("label", 0)))
    return keys, x, y


def train_arm(records_path: str | Path, out_path: str | Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in Path(records_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("No ARM records provided")

    keys, x, y = _vectorize(rows)

    if LogisticRegression is None:
        # Fallback: persist feature schema only.
        payload = {
            "model": "heuristic",
            "feature_keys": keys,
            "weights": [0.0 for _ in keys],
            "bias": 0.0,
            "calibrator": {"mode": "identity", "temperature": 1.0},
        }
        Path(out_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    x_np = np.asarray(x, dtype=float)
    y_np = np.asarray(y, dtype=int)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(x_np, y_np)
    logits = x_np @ clf.coef_[0] + clf.intercept_[0]
    p_raw = 1 / (1 + np.exp(-logits))

    if IsotonicRegression is not None:
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(p_raw, y_np)
        # Simple scalar proxy: choose temperature matching slope around 0.5.
        slope = 1.0
        if len(p_raw) > 3:
            idx = np.argsort(np.abs(p_raw - 0.5))[: min(20, len(p_raw))]
            if len(idx) > 1:
                pr = p_raw[idx]
                yr = iso.predict(pr)
                denom = max(pr.max() - pr.min(), 1e-6)
                slope = float((yr.max() - yr.min()) / denom)
        temperature = max(0.2, min(5.0, 1.0 / max(slope, 1e-6)))
    else:
        temperature = 1.0

    payload = {
        "model": "logistic",
        "feature_keys": keys,
        "weights": [float(v) for v in clf.coef_[0]],
        "bias": float(clf.intercept_[0]),
        "calibrator": {"mode": "temperature", "temperature": float(temperature)},
    }
    Path(out_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


if __name__ == "__main__":  # pragma: no cover
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("records")
    ap.add_argument("out")
    args = ap.parse_args()
    train_arm(args.records, args.out)

