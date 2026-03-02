from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class BudgetConfig:
    easy_seconds: float = 20.0
    medium_seconds: float = 60.0
    hard_seconds: float = 360.0
    default_token_budget: int = 1800
    sandbox_seconds_per_run: float = 2.0
    sandbox_max_runs: int = 4


@dataclass(slots=True)
class StopConfig:
    stop_pi_a: float = 0.92
    stop_pi_b: float = 0.88
    hard_stop_pi_a: float = 0.88
    hard_stop_pi_b: float = 0.84
    min_clusters_a: int = 1
    min_clusters_b: int = 2
    max_fragility: float = 0.45
    evi_stop_floor: float = 0.0


@dataclass(slots=True)
class AASConfig:
    flat_tau: float = 0.65
    small_margin: float = 0.08
    high_uncertainty: float = 0.25
    high_flip_rate: float = 0.30
    min_difficulty: float = 0.50
    sampling_temperature: float = 1.4
    cluster_bonus_scale: float = 0.15


@dataclass(slots=True)
class BeliefConfig:
    new_answer_log_prior: float = -13.815510557964274  # log(1e-6)
    other_log_prior: float = 0.0
    hard_refute_M: float = 20.0
    mild_refute_mu: float = 1.0
    weak_support_nu: float = 0.5


@dataclass(slots=True)
class EVIConfig:
    lambda_cost: float = 0.01
    eta_timeout: float = 0.2
    gain_min_per_sec: float = 0.002
    gain_streak_k: int = 3


@dataclass(slots=True)
class SystemConfig:
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    stop: StopConfig = field(default_factory=StopConfig)
    aas: AASConfig = field(default_factory=AASConfig)
    belief: BeliefConfig = field(default_factory=BeliefConfig)
    evi: EVIConfig = field(default_factory=EVIConfig)


DEFAULT_CONFIG = SystemConfig()
