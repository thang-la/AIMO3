from __future__ import annotations

import time
from dataclasses import dataclass, field

from aimo3.config import DifficultyBudgetConfig, SolverConfig
from aimo3.models import Difficulty, RouteDecision


@dataclass
class BudgetManager:
    time_limit_s: float
    max_attempts: int
    max_output_tokens: int
    max_tool_runs: int
    started_at: float = field(default_factory=time.monotonic)
    attempts_used: int = 0
    tool_runs_used: int = 0

    def remaining_time(self) -> float:
        return max(0.0, self.time_limit_s - (time.monotonic() - self.started_at))

    def can_attempt(self) -> bool:
        return self.attempts_used < self.max_attempts and self.remaining_time() > 0

    def use_attempt(self, count: int = 1) -> None:
        self.attempts_used += max(0, count)

    def can_run_tool(self) -> bool:
        return self.tool_runs_used < self.max_tool_runs and self.remaining_time() > 0

    def use_tool_run(self, count: int = 1) -> None:
        self.tool_runs_used += max(0, count)


def _pick_config(route: RouteDecision, config: SolverConfig) -> DifficultyBudgetConfig:
    if route.difficulty == Difficulty.EASY:
        return config.easy
    if route.difficulty == Difficulty.MEDIUM:
        return config.medium
    return config.hard


def allocate_budget(route: RouteDecision, config: SolverConfig) -> BudgetManager:
    picked = _pick_config(route, config)
    return BudgetManager(
        time_limit_s=picked.time_limit_s,
        max_attempts=picked.max_attempts,
        max_output_tokens=picked.max_output_tokens,
        max_tool_runs=picked.tool_runs,
    )
