from __future__ import annotations

from dataclasses import dataclass, field

from .types import Candidate, Evidence


@dataclass(slots=True)
class Hypothesis:
    id: int
    tag: str
    assumptions: list[str] = field(default_factory=list)
    artifacts: dict = field(default_factory=dict)
    tests: list[dict] = field(default_factory=list)
    weight: float = 0.0
    proposed_answers: list[tuple[int, float]] = field(default_factory=list)


HYP_TAGS_BY_DOMAIN = {
    "number_theory": ["NT_LTE", "NT_CRT", "NT_ORDER"],
    "algebra": ["ALG_SYMMETRY", "ALG_FUNC_EQ", "ALG_POLY"],
    "combinatorics": ["COMB_IE", "COMB_INVARIANT", "COMB_RECURRENCE"],
    "geometry": ["GEO_COORD", "GEO_COMPLEX", "GEO_POWERPOINT"],
    "mixed": ["MIXED_INVARIANT", "MIXED_MODEL", "MIXED_CONTRADICTION"],
}


def propose_hypothesis_tags(meta: dict, k: int) -> list[str]:
    tags = HYP_TAGS_BY_DOMAIN.get(meta.get("domain", "mixed"), HYP_TAGS_BY_DOMAIN["mixed"])
    out: list[str] = []
    while len(out) < k:
        out.extend(tags)
    return out[:k]


def init_hypotheses(meta: dict, k: int) -> dict[int, Hypothesis]:
    tags = propose_hypothesis_tags(meta, k=k)
    return {i: Hypothesis(id=i, tag=tag) for i, tag in enumerate(tags)}


def hypothesis_step(st, hyp: Hypothesis, op: str) -> list[Evidence]:
    if op == "LEMMA":
        strength = 0.8 if "INVARIANT" in hyp.tag or "CRT" in hyp.tag else 0.5
        ev = Evidence(type="hyp_lemma", target="OTHER", logbf=strength, weight=0.2, payload={"hyp_id": hyp.id})
        return [ev]

    if op == "MODEL":
        # MODEL action itself is executed in controller by invoking policy generation.
        ev = Evidence(type="hyp_model", target="OTHER", logbf=0.2, weight=0.1, payload={"hyp_id": hyp.id})
        return [ev]

    if op == "BRUTE_SMALL":
        ev = Evidence(type="hyp_brute", target="OTHER", logbf=0.4, weight=0.2, payload={"hyp_id": hyp.id})
        return [ev]

    if op == "CONTRADICTION":
        top_answer = max(st.pi.items(), key=lambda kv: kv[1])[0]
        if top_answer != "OTHER":
            ev = Evidence(type="hyp_contradiction", target=top_answer, logbf=-1.2, weight=0.35, payload={"hyp_id": hyp.id})
            return [ev]
        return []

    if op == "SWITCH_REP":
        ev = Evidence(type="hyp_switch", target="OTHER", logbf=0.2, weight=0.1, payload={"hyp_id": hyp.id})
        return [ev]

    return []


def apply_hypothesis_evidence(st, hyp: Hypothesis, evs: list[Evidence]) -> None:
    for ev in evs:
        if ev.type.startswith("hyp_"):
            hyp.weight += ev.weight * ev.logbf

