from __future__ import annotations

import math
from collections import defaultdict

from .ced import answer_cluster_support
from .types import Candidate
from .utils import clamp, softmax_dict


def baa_posterior(
    cands: list[Candidate],
    arm_ps: list[float],
    rho: list[list[float]],
    clusters: dict[int, list[int]],
) -> dict[int | str, float]:
    if not cands:
        return {"OTHER": 1.0}

    w: list[float] = []
    n = len(cands)
    for i in range(n):
        denom = 1.0 + sum(rho[i][j] for j in range(n) if j != i)
        w.append(1.0 / max(denom, 1.0))

    s: dict[int, float] = defaultdict(float)
    for i, c in enumerate(cands):
        p = clamp(arm_ps[i], 1e-6, 1 - 1e-6)
        s[c.answer] += w[i] * math.log(p / (1.0 - p))

    prior: dict[int, float] = defaultdict(float)
    support = answer_cluster_support(cands, clusters)
    for ans, k in support.items():
        prior[ans] = 0.3 * min(max(k - 1, 0), 3)

    logits = {ans: s[ans] + prior[ans] for ans in s}
    pi = softmax_dict(logits)

    pi_other = adaptive_other_mass(arm_ps, support)
    out = {k: (1.0 - pi_other) * v for k, v in pi.items()}
    out["OTHER"] = pi_other
    return out


def adaptive_other_mass(arm_ps: list[float], cluster_support: dict[int, int]) -> float:
    if not arm_ps:
        return 0.8
    low = sum(1 for p in arm_ps if p < 0.45) / len(arm_ps)
    weak_cluster = 0.0
    if cluster_support:
        weak_cluster = sum(1 for c in cluster_support.values() if c <= 1) / len(cluster_support)
    return clamp(0.01 + 0.2 * low + 0.1 * weak_cluster, 0.01, 0.45)

