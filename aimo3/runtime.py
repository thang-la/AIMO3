from __future__ import annotations

import os
from dataclasses import replace

from aimo3.config import LLMRuntimeConfig, SolverConfig


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    return int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    return float(raw)


def solver_config_from_env(base: SolverConfig | None = None) -> SolverConfig:
    cfg = base or SolverConfig()
    llm: LLMRuntimeConfig = cfg.llm
    llm = replace(
        llm,
        backend=os.getenv("AIMO3_BACKEND", llm.backend),
        model_main=os.getenv("AIMO3_MODEL_MAIN", llm.model_main),
        model_fast=os.getenv("AIMO3_MODEL_FAST", llm.model_fast),
        tensor_parallel_size=_env_int("AIMO3_TENSOR_PARALLEL_SIZE", llm.tensor_parallel_size),
        gpu_memory_utilization=_env_float("AIMO3_GPU_MEMORY_UTILIZATION", llm.gpu_memory_utilization),
        max_model_len=_env_int("AIMO3_MAX_MODEL_LEN", llm.max_model_len),
        temperature_easy=_env_float("AIMO3_TEMP_EASY", llm.temperature_easy),
        temperature_medium=_env_float("AIMO3_TEMP_MEDIUM", llm.temperature_medium),
        temperature_hard=_env_float("AIMO3_TEMP_HARD", llm.temperature_hard),
    )
    return replace(
        cfg,
        llm=llm,
        enforce_real_backend=_env_bool("AIMO3_ENFORCE_REAL_BACKEND", cfg.enforce_real_backend),
        allow_demo_fallback=_env_bool("AIMO3_ALLOW_DEMO_FALLBACK", cfg.allow_demo_fallback),
    )
