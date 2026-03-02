from __future__ import annotations

from .esmp import is_split, need_refute
from .policies import policy_applicable
from .utils import top2


def seed_action_set(st) -> list[dict]:
    actions: list[dict] = []
    program = st.run_config.get("program", "A")
    if program == "A":
        actions.append({"type": "GEN", "policy": "A_TOOLFORMAL", "n": 1})
        actions.append({"type": "GEN", "policy": "A_SYMBOLIC", "n": 1})
    else:
        actions.append({"type": "GEN", "policy": "B_INVARIANT", "n": 1})
        actions.append({"type": "GEN", "policy": "B_REFUTE", "n": 1})

    d = st.meta.get("domain")
    if d == "geometry":
        actions.append({"type": "GEN", "policy": "GEO_COORD", "n": 1})
    if d in {"combinatorics", "number_theory", "mixed"}:
        actions.append({"type": "GEN", "policy": "COMB_SMALLBRUTE", "n": 1})
    return actions


def top_candidate_ids(st, k: int = 3) -> list[int]:
    ranked = sorted(st.candidates, key=lambda c: c.arm_p, reverse=True)
    return [c.id for c in ranked[:k]]


def enumerate_actions(st) -> list[dict]:
    acts: list[dict] = []

    for pol in st.run_config.get("allowed_policies", []):
        if policy_applicable(pol, st.meta, st.diagnostics):
            acts.append({"type": "GEN", "policy": pol, "n": 1})

    by_id = {c.id: c for c in st.candidates}
    for cid in top_candidate_ids(st, k=3):
        cand = by_id.get(cid)
        if cand and not cand.verifier.get("deep_done", False):
            acts.append({"type": "VERIFY", "cand_id": cid, "level": "deep"})

    a1, _ = max(st.pi.items(), key=lambda kv: kv[1])
    if a1 != "OTHER" and need_refute(st, a1):
        acts.append({"type": "REFUTE", "answer": int(a1), "mode": st.run_config.get("refute_mode", "mild")})

    (ta, _), (tb, _) = top2(st.pi)
    if ta != "OTHER" and tb != "OTHER" and is_split(st, ta, tb):
        acts.append({"type": "DISAMBIGUATE", "a": int(ta), "b": int(tb)})

    if st.diagnostics.get("misparse_risk", 0.0) > 0.6:
        acts.append({"type": "REPARSE", "strictness": "strict"})

    if st.diagnostics.get("hardness", 0.0) > 0.75 and st.diagnostics.get("cluster_diversity", 0) < 2:
        if not st.diagnostics.get("hyp_started"):
            acts.append({"type": "HYP_START", "k": st.run_config.get("hyp_k", 3)})
        else:
            for hyp_id in select_active_hypotheses(st, k=2):
                for op in ["LEMMA", "MODEL", "BRUTE_SMALL", "SWITCH_REP", "CONTRADICTION"]:
                    acts.append({"type": "HYP_STEP", "hyp_id": hyp_id, "op": op})

    return dedupe_actions(acts)


def select_active_hypotheses(st, k: int = 2) -> list[int]:
    hyps = st.diagnostics.get("hypotheses", {})
    ranked = sorted(hyps.values(), key=lambda h: h.weight if hasattr(h, "weight") else h.get("weight", 0.0), reverse=True)
    out = []
    for h in ranked[:k]:
        out.append(h.id if hasattr(h, "id") else h["id"])
    return out


def dedupe_actions(actions: list[dict]) -> list[dict]:
    seen = set()
    out = []
    for a in actions:
        key = tuple(sorted(a.items()))
        if key not in seen:
            seen.add(key)
            out.append(a)
    return out

