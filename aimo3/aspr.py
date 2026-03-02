from __future__ import annotations

from .policies import CandidateFactory, run_policy_generate
from .types import Candidate, RefuteEvent


def generate_perturbations(meta: dict, k: int = 4) -> list[dict]:
    perturbations = []
    mod = meta.get("modulus")
    for i in range(k):
        p = dict(meta)
        if mod:
            p["modulus"] = max(2, int(mod) + ((-1) ** i) * (i + 1))
        p["perturbation_id"] = i
        perturbations.append(p)
    return perturbations


def adversarial_second_pass(
    model,
    factory: CandidateFactory,
    problem: str,
    meta: dict,
    run_config: dict,
    diagnostics: dict,
    base_seed: int,
) -> list[Candidate]:
    cands: list[Candidate] = []
    for pmeta in generate_perturbations(meta, k=4):
        cands.extend(
            run_policy_generate(
                model,
                factory,
                "B_REFUTE",
                problem,
                pmeta,
                n=1,
                base_seed=base_seed + pmeta["perturbation_id"],
            )
        )

    if meta.get("domain") in {"combinatorics", "number_theory"}:
        cands.extend(
            run_policy_generate(
                model,
                factory,
                "COMB_SMALLBRUTE",
                problem,
                meta,
                n=1,
                base_seed=base_seed + 91,
            )
        )

    return cands


def map_refutation_outcome_to_event(
    target_answer: int,
    outcome: str,
    reused_corr: float,
    flags: list[str] | None = None,
) -> RefuteEvent:
    return RefuteEvent(
        target_answer=target_answer,
        outcome=outcome,
        reused_cluster_corr=reused_corr,
        concern_flags=flags or [],
    )


def run_refutation(st, answer: int, mode: str = "mild") -> RefuteEvent:
    # Heuristic refutation oracle; production hooks can replace this function.
    frag = st.diagnostics.get("fragility", {}).get(answer, 0.7)
    support = st.diagnostics.get("cluster_support", {}).get(answer, 1)
    reused_corr = 0.5 if support <= 1 else 0.2

    if mode == "strong" and frag > 0.6:
        return map_refutation_outcome_to_event(answer, "FOUND_CONTRADICTION", reused_corr, ["fragile_top"])
    if frag > 0.75:
        return map_refutation_outcome_to_event(answer, "FOUND_COUNTEREXAMPLE", reused_corr, ["high_fragility"])
    if frag > 0.4:
        return map_refutation_outcome_to_event(answer, "INCONCLUSIVE", reused_corr, ["partial_warning"])
    return map_refutation_outcome_to_event(answer, "FAILED_TO_REFUTE", reused_corr, [])

