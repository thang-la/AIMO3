from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Budget:
    total_seconds: float
    token_budget: int
    sandbox_runs_left: int
    started_at: float = field(default_factory=time.monotonic)

    def time_spent(self) -> float:
        return time.monotonic() - self.started_at

    def time_left(self) -> float:
        return max(self.total_seconds - self.time_spent(), 0.0)

    def consume_tokens(self, n: int) -> None:
        self.token_budget = max(0, self.token_budget - max(0, n))

    def consume_sandbox_run(self) -> None:
        self.sandbox_runs_left = max(0, self.sandbox_runs_left - 1)


@dataclass(slots=True)
class Candidate:
    id: int
    answer: int
    policy_id: str
    path_type: str
    trace: str
    trace_summary: str
    tool_code: str | None = None
    tool_artifacts: dict[str, Any] | None = None
    verifier: dict[str, Any] = field(default_factory=dict)
    arm_logbf: float = 0.0
    arm_p: float = 0.5
    arm_u: float = 1.0
    failure_logits: dict[str, float] = field(default_factory=dict)
    cluster_id: int = -1
    redundancy_w: float = 1.0


@dataclass(slots=True)
class Evidence:
    type: str
    target: int | str | tuple[int, int]
    logbf: float
    weight: float = 1.0
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Cluster:
    cluster_id: int
    member_ids: list[int] = field(default_factory=list)
    cluster_mode: str = "unknown"
    cluster_weight: float = 1.0
    top_lemmas: list[str] = field(default_factory=list)
    tool_signature: str = ""


@dataclass(slots=True)
class BeliefState:
    s: dict[int | str, float]
    pi: dict[int | str, float]
    u: dict[int | str, float]
    clusters: dict[int, Cluster]
    candidates: list[Candidate]
    meta: dict[str, Any]
    constraints: dict[str, Any]
    run_config: dict[str, Any]
    budget: Budget
    diagnostics: dict[str, Any]
    action_history: list[dict[str, Any]]


@dataclass(slots=True)
class RefuteEvent:
    target_answer: int
    outcome: str
    concern_flags: list[str] = field(default_factory=list)
    reused_cluster_corr: float = 0.0
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DisambiguationEvent:
    a: int
    b: int
    logbf_ab: float
    reliability: float
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class VerificationResult:
    answer: int
    hard_ok: bool = False
    shallow_ok: bool = False
    deep_ok: bool = False
    symbolic_ok: bool = False
    random_ok_rate: float = 0.0
    judge_prob: float = 0.0
    tool_ok: bool = False
    tool_timeout: bool = False
    tool_error: bool = False
    contradictions: int = 0
    flags: list[str] = field(default_factory=list)
    artifacts: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ARMOutput:
    logbf_support: float
    p_correct: float
    uncertainty: float
    failure_logits: dict[str, float]


@dataclass(slots=True)
class PairwiseCorrelation:
    i: int
    j: int
    rho: float
