from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from .types import ARMOutput, Candidate, VerificationResult
from .utils import clamp, sigmoid


@dataclass(slots=True)
class ARMCalibrator:
    mode: str = "identity"  # identity | temperature
    temperature: float = 1.0

    def __call__(self, p_raw: float) -> float:
        p = clamp(p_raw, 1e-6, 1.0 - 1e-6)
        if self.mode == "temperature":
            logit = math.log(p / (1.0 - p))
            t = max(self.temperature, 1e-6)
            return sigmoid(logit / t)
        return p


@dataclass(slots=True)
class ARMModel:
    calibrator: ARMCalibrator = field(default_factory=ARMCalibrator)

    def predict(self, features: dict[str, Any]) -> ARMOutput:
        # Linear logit baseline using explicit feature schema from spec.
        logit = 0.0
        logit += 1.8 * float(features.get("hard_constraints_pass", 0.0))
        logit += 0.8 * float(features.get("sympy_simplify_success", 0.0))
        logit += 0.6 * float(features.get("python_exec_success", 0.0))
        logit += 0.6 * float(features.get("python_output_stability", 0.0))
        logit += 0.8 * float(features.get("random_test_pass_rate", 0.0))
        logit += 0.7 * float(features.get("judge_prob_correct", 0.0))
        logit += 0.5 * float(features.get("vote_share_of_answer", 0.0))
        logit += 0.6 * float(features.get("top_answer_cluster_support", 0.0))
        logit += 0.3 * float(features.get("alternate_method_agreement", 0.0))
        logit -= 0.8 * float(features.get("perturbation_flip_rate", 0.0))
        logit -= 0.6 * float(features.get("self_contradiction_flags", 0.0))
        logit -= 0.5 * float(features.get("uses_floats", 0.0))

        p_raw = sigmoid(logit)
        p_cal = self.calibrator(p_raw)
        p_cal = clamp(p_cal, 1e-6, 1 - 1e-6)

        prior = clamp(float(features.get("base_rate_prior", 0.2)), 1e-6, 1 - 1e-6)
        logbf = math.log(p_cal / (1.0 - p_cal)) - math.log(prior / (1.0 - prior))

        variance_proxy = 0.0
        variance_proxy += 0.5 * (1.0 - float(features.get("python_output_stability", 0.0)))
        variance_proxy += 0.5 * float(features.get("perturbation_flip_rate", 0.0))
        variance_proxy += 0.3 * (1.0 - float(features.get("top_answer_cluster_support", 0.0)))
        variance_proxy = clamp(variance_proxy, 0.0, 1.0)

        failure_logits = {
            "arith": float(features.get("uses_floats", 0.0)) + 0.2 * float(features.get("python_runtime_ms", 0.0)) / 5000.0,
            "missing_case": float(features.get("perturbation_flip_rate", 0.0)),
            "geometry_blind": float(features.get("geometry_indicator", 0.0)) * (1.0 - float(features.get("sympy_simplify_success", 0.0))),
            "misparse": float(features.get("num_constraints_extracted", 0.0) == 0),
            "tool_unstable": 1.0 - float(features.get("python_output_stability", 0.0)),
        }
        return ARMOutput(
            logbf_support=logbf,
            p_correct=p_cal,
            uncertainty=variance_proxy,
            failure_logits=failure_logits,
        )


def build_arm_features(
    meta: dict,
    cand: Candidate,
    verify: VerificationResult,
    all_candidates: list[Candidate],
    cluster_support: dict[int, int],
) -> dict[str, Any]:
    same_answer = [c for c in all_candidates if c.answer == cand.answer]
    unique_answer_count = len({c.answer for c in all_candidates})
    vote_share = len(same_answer) / max(len(all_candidates), 1)

    sandbox = (verify.artifacts or {}).get("sandbox", {})
    runtime_ms = float(sandbox.get("runtime_ms", 0.0))
    stability = float(sandbox.get("deterministic_score", 0.0)) if sandbox else 0.0

    uses_floats = 0.0
    if cand.tool_code:
        uses_floats = 1.0 if "float" in cand.tool_code or "." in cand.tool_code else 0.0

    features = {
        "hard_constraints_pass": float(verify.hard_ok),
        "num_constraints_extracted": float(meta.get("num_constraints_extracted", 0) or 0),
        "constraint_tightness": 1.0 if meta.get("modulus") else 0.4,
        "sympy_simplify_success": float(verify.symbolic_ok),
        "num_random_tests": 6.0,
        "random_test_pass_rate": float(verify.random_ok_rate),
        "judge_prob_correct": float(getattr(verify, "judge_prob", 0.0)),
        "z3_sat_consistency": 0.5,
        "python_exec_success": float(verify.tool_ok),
        "python_runtime_ms": runtime_ms,
        "python_output_stability": stability,
        "ast_complexity": float(len(cand.tool_code or "")),
        "uses_floats": uses_floats,
        "float_usage_count": uses_floats,
        "unique_answer_count": float(unique_answer_count),
        "vote_share_of_answer": float(vote_share),
        "path_diversity": float(len({c.path_type for c in all_candidates})) / 4.0,
        "cluster_count": float(len({c.cluster_id for c in all_candidates if c.cluster_id >= 0})),
        "top_answer_cluster_support": float(cluster_support.get(cand.answer, 1)),
        "avg_token_logprob": 0.0,
        "answer_margin": 0.0,
        "trace_length_tokens": float(len((cand.trace or "").split())),
        "self_contradiction_flags": float(verify.contradictions),
        "perturbation_flip_rate": 0.0,
        "alternate_method_agreement": float(cluster_support.get(cand.answer, 1) > 1),
        "domain": meta.get("domain", "mixed"),
        "difficulty": float(meta.get("difficulty", 0.5)),
        "geometry_indicator": float(meta.get("domain") == "geometry"),
        "modulus_present": float(meta.get("modulus") is not None),
        "base_rate_prior": base_rate_prior(cand.policy_id, meta),
    }
    return features


def base_rate_prior(policy_id: str, meta: dict) -> float:
    d = meta.get("difficulty", 0.5)
    if policy_id in {"A_TOOLFORMAL", "A_SYMBOLIC"}:
        base = 0.32
    elif policy_id in {"B_INVARIANT", "B_REFUTE"}:
        base = 0.22
    elif policy_id == "GEO_COORD":
        base = 0.28
    else:
        base = 0.18
    return clamp(base * (1.0 - 0.45 * d), 0.05, 0.6)
