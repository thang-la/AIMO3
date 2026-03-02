from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class CandidateRecord:
    problem_id: str
    candidate_id: int
    features: dict[str, float]
    label: int
    failure_type: str
    policy_id: str
    answer: int


@dataclass(slots=True)
class PairRecord:
    problem_id: str
    i: int
    j: int
    pair_features: dict[str, float]
    p_i: float
    p_j: float
    y_i: int
    y_j: int


@dataclass(slots=True)
class DecisionStepRecord:
    problem_id: str
    t: int
    state_features: dict[str, float]
    action: dict[str, Any]
    outcome: dict[str, Any]
    delta_v: float
    cost: float
    timeout: int


@dataclass(slots=True)
class RolloutRecord:
    problem_id: str
    run_program: str
    candidate_records: list[CandidateRecord] = field(default_factory=list)
    pair_records: list[PairRecord] = field(default_factory=list)
    decision_steps: list[DecisionStepRecord] = field(default_factory=list)

