from __future__ import annotations

import math
from collections import defaultdict

from .config import SystemConfig
from .types import BeliefState, Candidate, DisambiguationEvent, Evidence, RefuteEvent
from .utils import clamp, normalize_dict, softmax_dict, top1, top2


def init_belief_state_scores(cfg: SystemConfig) -> tuple[dict[int | str, float], dict[int | str, float], dict[int | str, float]]:
    s = {"OTHER": cfg.belief.other_log_prior}
    pi = {"OTHER": 1.0}
    u = {"OTHER": 1.0}
    return s, pi, u


def ensure_answer_present(st: BeliefState, answer: int, cfg: SystemConfig) -> None:
    if answer not in st.s:
        st.s[answer] = cfg.belief.new_answer_log_prior
    if answer not in st.u:
        st.u[answer] = 1.0


def apply_evidence(st: BeliefState, ev: Evidence) -> None:
    target = ev.target
    if isinstance(target, tuple):
        a, b = target
        st.s[a] = st.s.get(a, 0.0) + ev.weight * ev.logbf
        st.s[b] = st.s.get(b, 0.0) - ev.weight * ev.logbf
    else:
        st.s[target] = st.s.get(target, 0.0) + ev.weight * ev.logbf


def update_posterior(st: BeliefState) -> None:
    st.pi = softmax_dict(st.s)


def integrate_candidate_support(st: BeliefState, cand: Candidate, cfg: SystemConfig) -> None:
    ensure_answer_present(st, cand.answer, cfg)
    ev = Evidence(
        type="support",
        target=cand.answer,
        logbf=cand.arm_logbf,
        weight=cand.redundancy_w,
        payload={"candidate_id": cand.id},
    )
    apply_evidence(st, ev)


def integrate_hard_constraint_refutation(st: BeliefState, cand: Candidate, cfg: SystemConfig) -> None:
    ensure_answer_present(st, cand.answer, cfg)
    ev = Evidence(
        type="constraint",
        target=cand.answer,
        logbf=-cfg.belief.hard_refute_M,
        weight=1.0,
        payload={"candidate_id": cand.id, "reason": "hard_constraint_fail"},
    )
    apply_evidence(st, ev)


def integrate_refutation_online(st: BeliefState, ev: RefuteEvent, cfg: SystemConfig) -> None:
    if ev.outcome in {"FOUND_COUNTEREXAMPLE", "FOUND_CONTRADICTION"}:
        logbf = -cfg.belief.hard_refute_M
    elif ev.outcome == "INCONCLUSIVE":
        logbf = -cfg.belief.mild_refute_mu
    else:  # FAILED_TO_REFUTE
        logbf = cfg.belief.weak_support_nu
    weight = 1.0 - clamp(ev.reused_cluster_corr, 0.0, 0.95)
    apply_evidence(
        st,
        Evidence(
            type="refute",
            target=ev.target_answer,
            logbf=logbf,
            weight=weight,
            payload={"flags": ev.concern_flags, **ev.payload},
        ),
    )


def integrate_disambiguation_online(st: BeliefState, ev: DisambiguationEvent) -> None:
    w = clamp(ev.reliability, 0.0, 1.0)
    apply_evidence(
        st,
        Evidence(type="disambiguate", target=(ev.a, ev.b), logbf=ev.logbf_ab, weight=w, payload=ev.payload),
    )


def compute_baa_posterior(st: BeliefState, pi_other_override: float | None = None) -> dict[int | str, float]:
    # BAA posterior from current scores; already correlation-weighted in updates.
    base = softmax_dict(st.s)
    if pi_other_override is not None:
        other = clamp(pi_other_override, 1e-6, 0.8)
        scaled = {k: v for k, v in base.items() if k != "OTHER"}
        scaled = normalize_dict(scaled) if scaled else {}
        out = {k: (1.0 - other) * v for k, v in scaled.items()}
        out["OTHER"] = other
        return out
    return base


def adaptive_other_mass_from_state(st: BeliefState) -> float:
    if not st.candidates:
        return 0.75
    low_conf = [c for c in st.candidates if c.arm_p < 0.45]
    flat = 1.0
    if len(st.pi) > 1:
        (a1, p1), (_, p2) = top2(st.pi)
        del a1
        flat = 1.0 - clamp(p1 - p2, 0.0, 1.0)
    return clamp(0.02 + 0.25 * (len(low_conf) / len(st.candidates)) + 0.15 * flat, 0.01, 0.45)


def update_uncertainty_by_answer(st: BeliefState) -> None:
    num = defaultdict(float)
    den = defaultdict(float)
    for c in st.candidates:
        w = max(c.redundancy_w, 1e-6)
        num[c.answer] += w * c.arm_u
        den[c.answer] += w
    for ans, d in den.items():
        st.u[ans] = num[ans] / max(d, 1e-6)


def cluster_support_map(st: BeliefState) -> dict[int, int]:
    support: dict[int, set[int]] = defaultdict(set)
    for c in st.candidates:
        if c.cluster_id >= 0:
            support[c.answer].add(c.cluster_id)
    return {a: len(ids) for a, ids in support.items()}


def fragility_map(st: BeliefState) -> dict[int, float]:
    support = cluster_support_map(st)
    frag: dict[int, float] = {}
    for ans, count in support.items():
        u = st.u.get(ans, 1.0)
        frag[ans] = clamp(0.6 * u + 0.4 * (1.0 / max(count, 1)), 0.0, 1.0)
    for c in st.candidates:
        frag.setdefault(c.answer, clamp(st.u.get(c.answer, 1.0), 0.0, 1.0))
    return frag


def top_answers(st: BeliefState) -> tuple[tuple[int | str, float], tuple[int | str, float]]:
    return top2(st.pi)


def expected_correctness_if_stop(st: BeliefState) -> float:
    _, p = top1(st.pi)
    return p

