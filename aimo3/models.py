from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Domain(str, Enum):
    ALGEBRA = "algebra"
    NUMBER_THEORY = "number_theory"
    COMBINATORICS = "combinatorics"
    GEOMETRY = "geometry"
    MIXED = "mixed"


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class PathType(str, Enum):
    P0_SYMBOLIC = "P0_symbolic"
    P1_TOOL = "P1_tool"
    P2_REASONING = "P2_reasoning"
    P3_BACKSOLVE = "P3_backsolve"
    HARD_MODE = "hard_mode"


class PromptProgram(str, Enum):
    A = "A_tool_heavy"
    B = "B_theorem_heavy"


@dataclass
class ProblemMetadata:
    pid: str
    raw_text: str
    normalized_text: str
    statement_hash: str
    modulus: int | None = None
    extracted_equations: list[str] = field(default_factory=list)
    numbers: list[int] = field(default_factory=list)
    variables: list[str] = field(default_factory=list)
    detected_domain_hints: list[str] = field(default_factory=list)


@dataclass
class RouteDecision:
    domain: Domain
    difficulty: Difficulty
    difficulty_score: float
    requires_tool: bool
    try_symbolic_first: bool
    use_backsolve: bool
    allow_hard_mode: bool
    n_tool_attempts: int
    n_cot_attempts: int
    n_backsolve_attempts: int


@dataclass
class Candidate:
    path: PathType
    answer: int | None = None
    trace: str = ""
    python_code: str | None = None
    program: PromptProgram = PromptProgram.A
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class VerificationResult:
    candidate: Candidate
    normalized_answer: int | None = None
    hard_ok: bool = False
    symbolic_ok: bool = False
    random_ok: bool = False
    judge_prob: float = 0.0
    contradiction: bool = False
    sandbox_error: bool = False
    score: float = -999.0
    notes: list[str] = field(default_factory=list)
    artifacts: dict[str, Any] = field(default_factory=dict)
    self_consistency_vote_share: float = 0.0


@dataclass
class SolveResult:
    pid: str
    answer: int
    meta: ProblemMetadata
    route: RouteDecision
    candidates: list[VerificationResult]
    selected: VerificationResult | None
    reason: str

    def to_log_dict(self) -> dict[str, Any]:
        selected_answer = self.selected.normalized_answer if self.selected else None
        return {
            "pid": self.pid,
            "answer": self.answer,
            "selected_answer": selected_answer,
            "route": {
                "domain": self.route.domain.value,
                "difficulty": self.route.difficulty.value,
                "difficulty_score": self.route.difficulty_score,
            },
            "meta": {
                "hash": self.meta.statement_hash,
                "modulus": self.meta.modulus,
                "numbers": self.meta.numbers[:20],
                "equation_count": len(self.meta.extracted_equations),
            },
            "reason": self.reason,
            "candidates": [
                {
                    "path": c.candidate.path.value,
                    "answer": c.normalized_answer,
                    "score": c.score,
                    "hard_ok": c.hard_ok,
                    "symbolic_ok": c.symbolic_ok,
                    "random_ok": c.random_ok,
                    "judge_prob": c.judge_prob,
                    "vote_share": c.self_consistency_vote_share,
                    "notes": c.notes,
                }
                for c in self.candidates
            ],
        }
