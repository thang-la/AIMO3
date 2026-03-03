from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

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


def _looks_like_repo_id(model_ref: str) -> bool:
    return bool(model_ref and "/" in model_ref and not Path(model_ref).exists())


def _is_model_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    markers = (path / "config.json", path / "tokenizer.json", path / "tokenizer_config.json")
    return any(p.exists() for p in markers)


def _norm_name(s: str) -> str:
    return s.lower().replace("_", "-").replace(".", "-")


def _discover_kaggle_model_path(model_ref: str) -> str | None:
    root = Path(os.getenv("AIMO3_KAGGLE_MODEL_ROOT", "/kaggle/input"))
    if not root.exists():
        return None
    target = _norm_name(model_ref.split("/")[-1])
    candidates: list[Path] = []
    try:
        for p in root.rglob("*"):
            if not p.is_dir():
                continue
            if len(p.parts) - len(root.parts) > 5:
                continue
            name = _norm_name(p.name)
            if target in name or name in target:
                if _is_model_dir(p):
                    candidates.append(p)
    except Exception:
        return None
    if not candidates:
        return None
    candidates.sort(key=lambda p: (len(str(p)), str(p)))
    return str(candidates[0])


def _resolve_model_ref(model_ref: str) -> str:
    if not model_ref:
        return model_ref
    p = Path(model_ref)
    if p.exists():
        return str(p)
    if _looks_like_repo_id(model_ref):
        discovered = _discover_kaggle_model_path(model_ref)
        if discovered:
            return discovered
    return model_ref


def solver_config_from_env(base: SolverConfig | None = None) -> SolverConfig:
    cfg = base or SolverConfig()
    llm: LLMRuntimeConfig = cfg.llm
    llm = replace(
        llm,
        backend=os.getenv("AIMO3_BACKEND", llm.backend),
        model_main=_resolve_model_ref(os.getenv("AIMO3_MODEL_MAIN", llm.model_main)),
        model_fast=_resolve_model_ref(os.getenv("AIMO3_MODEL_FAST", llm.model_fast)),
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
        allow_reference_lookup=_env_bool("AIMO3_ALLOW_REFERENCE_LOOKUP", cfg.allow_reference_lookup),
        reference_similarity_threshold=_env_float(
            "AIMO3_REFERENCE_SIMILARITY_THRESHOLD", cfg.reference_similarity_threshold
        ),
        debug_enabled=_env_bool("AIMO3_DEBUG", cfg.debug_enabled),
        debug_include_raw_output=_env_bool("AIMO3_DEBUG_RAW_OUTPUT", cfg.debug_include_raw_output),
        debug_max_chars=_env_int("AIMO3_DEBUG_MAX_CHARS", cfg.debug_max_chars),
        debug_file_path=Path(os.getenv("AIMO3_DEBUG_FILE")) if os.getenv("AIMO3_DEBUG_FILE") else cfg.debug_file_path,
    )
