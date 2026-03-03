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
    path_diversity: float = 0.6
    validator_bonus: float = 0.8
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
            time_limit_s=90.0,
            max_attempts=8,
            max_output_tokens=2600,
            tool_runs=3,
        )
    )
    hard: DifficultyBudgetConfig = field(
        default_factory=lambda: DifficultyBudgetConfig(
            time_limit_s=360.0,
            max_attempts=16,
            max_output_tokens=6000,
            tool_runs=8,
        )
    )
    sandbox_timeout_s: float = 4.0
    sandbox_memory_mb: int = 1024
    confidence_threshold: float = 4.2
    vote_share_threshold: float = 0.5
    max_candidates: int = 80
    max_repair_rounds: int = 2
    repair_top_k: int = 3
    require_path_diversity_for_confidence: bool = True
    answer_upper_bound: int = 99999
    global_modulus_fallback: int = 100000
    run_log_dir: Path = field(default_factory=lambda: Path("runs"))
    enable_hard_mode: bool = True
    enforce_real_backend: bool = True
    allow_demo_fallback: bool = False
    allow_reference_lookup: bool = False
    reference_similarity_threshold: float = 0.985
    reference_path: Path = field(default_factory=lambda: Path("reference.csv"))
    debug_enabled: bool = False
    debug_include_raw_output: bool = False
    debug_max_chars: int = 1200
    debug_file_path: Path | None = None
    llm: LLMRuntimeConfig = field(default_factory=LLMRuntimeConfig)
