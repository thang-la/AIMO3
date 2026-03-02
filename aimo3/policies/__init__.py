from __future__ import annotations

import itertools
from dataclasses import dataclass

from ..models import BaseGeneratorModel
from ..types import Candidate
from ..utils import hash64


@dataclass(slots=True)
class PolicySpec:
    policy_id: str
    path_type: str
    temperature: float
    max_tokens: int
    prompt_template: str


POLICY_SPECS: dict[str, PolicySpec] = {
    "A_TOOLFORMAL": PolicySpec(
        policy_id="A_TOOLFORMAL",
        path_type="P1",
        temperature=0.0,
        max_tokens=900,
        prompt_template=(
            "Formalize constraints with exact arithmetic. Produce Python/SymPy model and compute final integer answer. "
            "Return ANSWER as int. Problem: {problem}"
        ),
    ),
    "A_SYMBOLIC": PolicySpec(
        policy_id="A_SYMBOLIC",
        path_type="P0",
        temperature=0.1,
        max_tokens=700,
        prompt_template=(
            "Use symbolic algebra/number theory transforms. Prioritize CRT/LTE/factor constraints. "
            "Return a precise integer answer. Problem: {problem}"
        ),
    ),
    "B_INVARIANT": PolicySpec(
        policy_id="B_INVARIANT",
        path_type="P2",
        temperature=0.3,
        max_tokens=900,
        prompt_template=(
            "Search invariants, bounds, monotonicity, parity. Produce alternate method independent from tool-first modeling. "
            "Problem: {problem}"
        ),
    ),
    "B_REFUTE": PolicySpec(
        policy_id="B_REFUTE",
        path_type="P3",
        temperature=0.3,
        max_tokens=1000,
        prompt_template=(
            "Attempt contradiction or counterexample against current dominant assumptions, then re-derive answer independently. "
            "Problem: {problem}"
        ),
    ),
    "GEO_COORD": PolicySpec(
        policy_id="GEO_COORD",
        path_type="P1",
        temperature=0.2,
        max_tokens=1000,
        prompt_template=(
            "Force coordinate/complex-plane formalization and numeric sanity checks for geometry. Problem: {problem}"
        ),
    ),
    "COMB_SMALLBRUTE": PolicySpec(
        policy_id="COMB_SMALLBRUTE",
        path_type="P3",
        temperature=0.2,
        max_tokens=800,
        prompt_template=(
            "Build small-instance brute-force validation and extrapolate only if pattern is stable. Problem: {problem}"
        ),
    ),
}


class CandidateFactory:
    def __init__(self) -> None:
        self._counter = itertools.count()

    def next_id(self) -> int:
        return next(self._counter)


def policy_applicable(policy_id: str, meta: dict, diagnostics: dict) -> bool:
    domain = meta.get("domain", "mixed")
    if policy_id == "GEO_COORD" and domain != "geometry":
        return False
    if policy_id == "COMB_SMALLBRUTE" and domain not in {"combinatorics", "mixed", "number_theory"}:
        return False
    if policy_id == "B_REFUTE" and diagnostics.get("cluster_diversity", 0) > 3 and diagnostics.get("fragile_top", 0.0) < 0.2:
        return False
    return True


def run_policy_generate(
    model: BaseGeneratorModel,
    factory: CandidateFactory,
    policy_id: str,
    problem: str,
    meta: dict,
    *,
    n: int,
    base_seed: int,
) -> list[Candidate]:
    spec = POLICY_SPECS[policy_id]
    candidates: list[Candidate] = []
    for k in range(max(1, n)):
        prompt = spec.prompt_template.format(problem=problem)
        seed = int(hash64(f"{base_seed}:{policy_id}:{k}:{meta.get('problem_hash','0')}") & 0x7FFFFFFF)
        gen = model.generate(
            prompt,
            seed=seed,
            max_tokens=spec.max_tokens,
            temperature=spec.temperature,
        )
        answer = int(gen.answer)
        if meta.get("modulus"):
            answer = answer % int(meta["modulus"])
        answer = answer % 100000
        cand = Candidate(
            id=factory.next_id(),
            answer=answer,
            policy_id=policy_id,
            path_type=spec.path_type,
            trace=gen.trace,
            trace_summary=_summarize_trace(gen.trace),
            tool_code=gen.tool_code,
            tool_artifacts={
                "prompt_hash": str(hash64(prompt)),
                "policy": policy_id,
                "temperature": spec.temperature,
            },
        )
        candidates.append(cand)
    return candidates


def _summarize_trace(trace: str) -> str:
    parts = trace.split(".")
    return ". ".join(parts[:2]).strip()

