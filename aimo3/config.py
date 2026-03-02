from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class DifficultyBudgetConfig:
    time_limit_s: float
    max_attempts: int
    max_output_tokens: int
    tool_runs: int


@dataclass(frozen=True)
class VerificationWeights:
    hard_constraints: float = 2.0
    symbolic_consistency: float = 1.5
    randomized_tests: float = 1.0
    judge_prob: float = 1.0
    self_consistency: float = 0.5
    contradiction_penalty: float = -2.0
    sandbox_penalty: float = -1.0


@dataclass(frozen=True)
class LLMRuntimeConfig:
    backend: str = "auto"  # auto|vllm|transformers|heuristic
    model_main: str = "openai/gpt-oss-120b"
    model_fast: str = "openai/gpt-oss-20b"
    max_new_tokens_easy: int = 900
    max_new_tokens_medium: int = 1300
    max_new_tokens_hard: int = 1800
    top_p: float = 0.95
    temperature_easy: float = 0.0
    temperature_medium: float = 0.2
    temperature_hard: float = 0.4
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.90
    max_model_len: int = 8192
    trust_remote_code: bool = True


@dataclass(frozen=True)
class SolverConfig:
    easy: DifficultyBudgetConfig = field(
        default_factory=lambda: DifficultyBudgetConfig(
            time_limit_s=20.0,
            max_attempts=2,
            max_output_tokens=1200,
            tool_runs=1,
        )
    )
    medium: DifficultyBudgetConfig = field(
        default_factory=lambda: DifficultyBudgetConfig(
            time_limit_s=60.0,
            max_attempts=5,
            max_output_tokens=2200,
            tool_runs=2,
        )
    )
    hard: DifficultyBudgetConfig = field(
        default_factory=lambda: DifficultyBudgetConfig(
            time_limit_s=240.0,
            max_attempts=10,
            max_output_tokens=4500,
            tool_runs=5,
        )
    )
    sandbox_timeout_s: float = 4.0
    sandbox_memory_mb: int = 1024
    confidence_threshold: float = 4.2
    vote_share_threshold: float = 0.5
    max_candidates: int = 80
    answer_upper_bound: int = 99999
    global_modulus_fallback: int = 100000
    run_log_dir: Path = field(default_factory=lambda: Path("runs"))
    enable_hard_mode: bool = True
    enforce_real_backend: bool = True
    allow_demo_fallback: bool = False
    allow_reference_lookup: bool = False
    reference_path: Path = field(default_factory=lambda: Path("reference.csv"))
    llm: LLMRuntimeConfig = field(default_factory=LLMRuntimeConfig)
