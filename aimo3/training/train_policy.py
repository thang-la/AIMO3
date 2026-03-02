from __future__ import annotations

import json
from pathlib import Path

try:  # pragma: no cover
    import numpy as np
    from sklearn.linear_model import LogisticRegression
except Exception:  # pragma: no cover
    np = None
    LogisticRegression = None


def _action_to_label(action: dict) -> str:
    t = action.get("type", "")
    if t == "GEN":
        return f"GEN:{action.get('policy','')}"
    if t == "HYP_STEP":
        return f"HYP_STEP:{action.get('op','')}"
    return t


def train_policy(records_path: str | Path, out_path: str | Path) -> dict:
    rows = [json.loads(line) for line in Path(records_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    labels = sorted({_action_to_label(r.get("action", {})) for r in rows})
    label_to_idx = {l: i for i, l in enumerate(labels)}

    feat_keys = sorted({k for r in rows for k in r.get("state_features", {}).keys()})
    x = [[float(r.get("state_features", {}).get(k, 0.0)) for k in feat_keys] for r in rows]
    y = [label_to_idx[_action_to_label(r.get("action", {}))] for r in rows]

    if LogisticRegression is None or np is None:
        payload = {"model": "heuristic", "feature_keys": feat_keys, "labels": labels}
        Path(out_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    x_np = np.asarray(x, dtype=float)
    y_np = np.asarray(y, dtype=int)
    clf = LogisticRegression(max_iter=1000, multi_class="multinomial")
    clf.fit(x_np, y_np)

    payload = {
        "model": "logreg_multinomial",
        "feature_keys": feat_keys,
        "labels": labels,
        "coef": [[float(v) for v in row] for row in clf.coef_],
        "bias": [float(v) for v in clf.intercept_],
    }
    Path(out_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


if __name__ == "__main__":  # pragma: no cover
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("records")
    ap.add_argument("out")
    args = ap.parse_args()
    train_policy(args.records, args.out)

