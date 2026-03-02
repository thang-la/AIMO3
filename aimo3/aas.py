from __future__ import annotations

import random

from .config import SystemConfig
from .utils import normalize_dict, top2


def should_sample(st, cfg: SystemConfig) -> bool:
    (a1, p1), (a2, p2) = top2(st.pi)
    if a1 == "OTHER":
        return False
    flat = (p1 < cfg.aas.flat_tau) and ((p1 - p2) < cfg.aas.small_margin)

    cluster_support = st.diagnostics.get("cluster_support", {}).get(a1, 1)
    flip_rate = st.diagnostics.get("perturb_flip_rate", 0.0)
    uncertainty = st.u.get(a1, 0.0)
    fragile = cluster_support == 1 or flip_rate > cfg.aas.high_flip_rate or uncertainty > cfg.aas.high_uncertainty

    return bool(flat and fragile and st.meta.get("difficulty", 0.0) > cfg.aas.min_difficulty)


def sampling_distribution(st, cfg: SystemConfig) -> dict[int | str, float]:
    q: dict[int | str, float] = {}
    for a, p in st.pi.items():
        if a == "OTHER":
            continue
        support = st.diagnostics.get("cluster_support", {}).get(a, 1)
        bonus = 1.0 + cfg.aas.cluster_bonus_scale * (support - 1)
        q[a] = (p ** (1.0 / cfg.aas.sampling_temperature)) * bonus
    return normalize_dict(q)


def select_answer(st, cfg: SystemConfig, rng: random.Random) -> tuple[int, dict]:
    (a1, p1), _ = top2(st.pi)
    if a1 == "OTHER":
        return 0, {"mode": "other_fallback"}

    if p1 >= st.run_config.get("stop_pi", 0.9) and st.diagnostics.get("cluster_support", {}).get(a1, 0) >= st.run_config.get("require_clusters", 1):
        return int(a1), {"mode": "deterministic_high_conf"}

    if should_sample(st, cfg):
        q = sampling_distribution(st, cfg)
        r = rng.random()
        c = 0.0
        for a, p in sorted(q.items(), key=lambda kv: kv[1], reverse=True):
            c += p
            if r <= c:
                return int(a), {"mode": "stochastic_ambiguous", "q": q}
        last = next(iter(q.keys()))
        return int(last), {"mode": "stochastic_ambiguous", "q": q}

    return int(a1), {"mode": "deterministic_default"}

