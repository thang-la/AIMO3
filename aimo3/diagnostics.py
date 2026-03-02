from __future__ import annotations

from collections import Counter

from .belief import cluster_support_map, fragility_map, top_answers
from .utils import safe_entropy


def init_diagnostics(meta: dict, constraints: dict, run_config: dict) -> dict:
    return {
        "cluster_support": {},
        "fragility": {},
        "blindspot_flags": {},
        "misparse_risk": 0.2 if constraints.get("num_constraints_extracted", 0) > 0 else 0.65,
        "hardness": float(meta.get("difficulty", 0.5)),
        "cluster_diversity": 0,
        "perturb_flip_rate": 0.0,
        "gain_rate": 0.0,
        "gain_rate_streak": 0,
        "pi": {"OTHER": 1.0},
        "run_program": run_config.get("program", "A"),
        "failure_type_top": "",
        "action_counts": Counter(),
        "hyp_started": False,
    }


def update_diagnostics(st) -> dict:
    support = cluster_support_map(st)
    frag = fragility_map(st)
    (a1, p1), (a2, p2) = top_answers(st)

    # marginal gain estimate from action history
    gain_rate = st.diagnostics.get("gain_rate", 0.0)
    streak = st.diagnostics.get("gain_rate_streak", 0)
    if st.action_history:
        last = st.action_history[-1]
        prev_top = last.get("pi_top_prev", last.get("pi_top", p1))
        now_top = p1
        dt = max(0.5, last.get("delta_t", 1.0))
        g = (now_top - prev_top) / dt
        if g < 0.002:
            streak += 1
        else:
            streak = 0
        gain_rate = g

    failure_type_top = infer_failure_type(st, a1)

    d = dict(st.diagnostics)
    d["cluster_support"] = support
    d["fragility"] = frag
    d["cluster_diversity"] = len(set(c.cluster_id for c in st.candidates if c.cluster_id >= 0))
    d["posterior_entropy"] = safe_entropy(st.pi.values()) if st.pi else 0.0
    d["split_margin"] = p1 - p2
    d["fragile_top"] = frag.get(a1, 1.0) if a1 != "OTHER" else 1.0
    d["pi"] = dict(st.pi)
    d["gain_rate"] = gain_rate
    d["gain_rate_streak"] = streak
    d["failure_type_top"] = failure_type_top
    d["blindspot_flags"] = {
        "geometry_no_formalization": st.meta.get("domain") == "geometry" and not any(c.policy_id == "GEO_COORD" for c in st.candidates),
        "single_cluster_top": support.get(a1, 0) <= 1 if a1 != "OTHER" else False,
        "misparse_risk": d.get("misparse_risk", 0.0) > 0.6,
    }
    return d


def infer_failure_type(st, a1: int | str) -> str:
    if a1 == "OTHER":
        return "misparse"
    if st.meta.get("domain") == "geometry" and not any(c.policy_id == "GEO_COORD" for c in st.candidates):
        return "geometry_blind"
    if st.diagnostics.get("misparse_risk", 0.0) > 0.6:
        return "misparse"
    top_cands = [c for c in st.candidates if c.answer == a1]
    if top_cands and any(c.failure_logits.get("missing_case", 0.0) > 0.6 for c in top_cands):
        return "missing_case"
    if top_cands and any(c.failure_logits.get("tool_unstable", 0.0) > 0.5 for c in top_cands):
        return "tool_unstable"
    return ""

