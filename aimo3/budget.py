from __future__ import annotations

from .config import SystemConfig
from .types import Budget


def allocate_budget(meta: dict, base_mode: dict, run_config: dict, cfg: SystemConfig) -> Budget:
    bucket = base_mode.get("difficulty_bucket", "medium")
    if bucket == "easy":
        total = cfg.budget.easy_seconds
        tokens = 1200
        sandbox = 1
    elif bucket == "hard":
        total = cfg.budget.hard_seconds
        tokens = 5000
        sandbox = cfg.budget.sandbox_max_runs
    else:
        total = cfg.budget.medium_seconds
        tokens = 2600
        sandbox = max(2, cfg.budget.sandbox_max_runs // 2)

    if run_config.get("program") == "B":
        total *= 1.05
        tokens = int(tokens * 1.1)

    return Budget(total_seconds=total, token_budget=tokens, sandbox_runs_left=sandbox)

