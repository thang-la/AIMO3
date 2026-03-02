from __future__ import annotations

from .config import SystemConfig
from .utils import hash32


def select_run_config(run_seed: int, problem_hash: str, meta: dict, base_mode: dict, cfg: SystemConfig) -> dict:
    bit = hash32(f"{run_seed}:{problem_hash}") & 1
    if bit == 0:
        return {
            "program": "A",
            "tool_priority": True,
            "adversarial": False,
            "allowed_policies": ["A_TOOLFORMAL", "A_SYMBOLIC", "GEO_COORD", "COMB_SMALLBRUTE"],
            "stop_pi": cfg.stop.stop_pi_a,
            "hard_stop_pi": cfg.stop.hard_stop_pi_a,
            "min_clusters": cfg.stop.min_clusters_a,
            "max_fragility": cfg.stop.max_fragility,
            "evi_stop_floor": cfg.stop.evi_stop_floor,
            "lambda_cost": cfg.evi.lambda_cost,
            "eta_timeout": cfg.evi.eta_timeout,
            "refute_mode": "mild",
            "hyp_k": 3,
            "require_clusters": cfg.stop.min_clusters_a,
        }
    return {
        "program": "B",
        "tool_priority": False,
        "adversarial": True,
        "allowed_policies": ["B_INVARIANT", "B_REFUTE", "GEO_COORD", "COMB_SMALLBRUTE"],
        "stop_pi": cfg.stop.stop_pi_b,
        "hard_stop_pi": cfg.stop.hard_stop_pi_b,
        "min_clusters": cfg.stop.min_clusters_b,
        "max_fragility": cfg.stop.max_fragility,
        "evi_stop_floor": cfg.stop.evi_stop_floor,
        "lambda_cost": cfg.evi.lambda_cost,
        "eta_timeout": cfg.evi.eta_timeout,
        "refute_mode": "strong",
        "hyp_k": 4,
        "require_clusters": cfg.stop.min_clusters_b,
    }


def adapt_run_config_for_low_diversity(run_config: dict, diagnostics: dict) -> dict:
    if diagnostics.get("cluster_diversity", 0) < 2:
        adjusted = dict(run_config)
        if adjusted.get("program") == "A":
            for pol in ["B_INVARIANT", "B_REFUTE"]:
                if pol not in adjusted["allowed_policies"]:
                    adjusted["allowed_policies"].append(pol)
        else:
            for pol in ["A_TOOLFORMAL", "A_SYMBOLIC"]:
                if pol not in adjusted["allowed_policies"]:
                    adjusted["allowed_policies"].append(pol)
        return adjusted
    return run_config

