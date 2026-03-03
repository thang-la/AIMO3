from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aimo3.config import LLMRuntimeConfig
from aimo3.models import (
    Candidate,
    Difficulty,
    PathType,
    ProblemMetadata,
    PromptProgram,
    RouteDecision,
    VerificationResult,
)


@dataclass
class NeuralJudge:
    """Heuristic judge that can be replaced by a trained model."""

    name: str = "heuristic-judge-v2"

    def score(self, problem: str, trace: str, answer: int, artifacts: dict) -> float:
        base = 0.45
        if artifacts.get("tool_ok"):
            base += 0.2
        if artifacts.get("validator_ok"):
            base += 0.2
        if "independent check" in trace.lower() or "cross-check" in trace.lower():
            base += 0.1
        if artifacts.get("sandbox_timeout"):
            base -= 0.2
        if answer < 0:
            base -= 0.5
        return float(min(1.0, max(0.0, base)))


class InferenceUnavailableError(RuntimeError):
    """Raised when runtime backend/model cannot be loaded."""


class BaseLLMBackend:
    name = "base"

    def generate_tool_candidates(
        self,
        problem: str,
        meta: ProblemMetadata,
        route: RouteDecision,
        n: int,
        seed: int,
        program: PromptProgram,
    ) -> list[Candidate]:
        raise NotImplementedError

    def generate_reasoning_candidates(
        self,
        problem: str,
        meta: ProblemMetadata,
        route: RouteDecision,
        n: int,
        seed: int,
        program: PromptProgram,
    ) -> list[Candidate]:
        raise NotImplementedError

    def generate_backsolve_candidates(
        self,
        problem: str,
        meta: ProblemMetadata,
        route: RouteDecision,
        n: int,
        seed: int,
        program: PromptProgram,
    ) -> list[Candidate]:
        raise NotImplementedError

    def generate_repair_candidates(
        self,
        problem: str,
        meta: ProblemMetadata,
        route: RouteDecision,
        n: int,
        seed: int,
        program: PromptProgram,
        prior_verified: list[VerificationResult],
    ) -> list[Candidate]:
        return []


class HeuristicLLMBackend(BaseLLMBackend):
    """Deterministic fallback backend used only when explicitly allowed."""

    name = "heuristic-backend-v2"

    @staticmethod
    def _hash_answer(problem: str, tag: str, seed: int, modulus: int | None) -> int:
        key = f"{problem}|{tag}|{seed}"
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        value = int(digest[:16], 16)
        if modulus is not None and modulus > 0:
            return value % modulus
        return value % 100000

    def _mk_trace(self, tag: str) -> str:
        return f"Fallback heuristic candidate via {tag}; use only for smoke testing."

    def generate_tool_candidates(
        self,
        problem: str,
        meta: ProblemMetadata,
        route: RouteDecision,
        n: int,
        seed: int,
        program: PromptProgram,
    ) -> list[Candidate]:
        out: list[Candidate] = []
        for i in range(n):
            answer = self._hash_answer(problem, f"tool-{program.value}-{i}", seed + i, meta.modulus)
            code = f"modulus = {meta.modulus or 100000}\nANSWER = {answer} % modulus\n"
            validator_code = "IS_VALID = isinstance(CANDIDATE_ANSWER, int)"
            out.append(
                Candidate(
                    path=PathType.P1_TOOL,
                    answer=None,
                    python_code=code,
                    trace=self._mk_trace("tool"),
                    program=program,
                    metadata={
                        "backend": self.name,
                        "attempt": i,
                        "validator_code": validator_code,
                        "confidence_hint": 0.2,
                    },
                )
            )
        return out

    def generate_reasoning_candidates(
        self,
        problem: str,
        meta: ProblemMetadata,
        route: RouteDecision,
        n: int,
        seed: int,
        program: PromptProgram,
    ) -> list[Candidate]:
        return [
            Candidate(
                path=PathType.P2_REASONING,
                answer=self._hash_answer(problem, f"reason-{i}", seed + i, meta.modulus),
                trace=self._mk_trace("reasoning"),
                program=program,
                metadata={"backend": self.name, "attempt": i, "confidence_hint": 0.1},
            )
            for i in range(n)
        ]

    def generate_backsolve_candidates(
        self,
        problem: str,
        meta: ProblemMetadata,
        route: RouteDecision,
        n: int,
        seed: int,
        program: PromptProgram,
    ) -> list[Candidate]:
        return [
            Candidate(
                path=PathType.P3_BACKSOLVE,
                answer=self._hash_answer(problem, f"backsolve-{i}", seed + i, meta.modulus),
                trace=self._mk_trace("backsolve"),
                program=program,
                metadata={"backend": self.name, "attempt": i, "confidence_hint": 0.1},
            )
            for i in range(n)
        ]

    def generate_repair_candidates(
        self,
        problem: str,
        meta: ProblemMetadata,
        route: RouteDecision,
        n: int,
        seed: int,
        program: PromptProgram,
        prior_verified: list[VerificationResult],
    ) -> list[Candidate]:
        return [
            Candidate(
                path=PathType.P4_REPAIR,
                answer=self._hash_answer(problem, f"repair-{i}", seed + i * 17, meta.modulus),
                trace=self._mk_trace("repair"),
                program=program,
                metadata={"backend": self.name, "attempt": i, "confidence_hint": 0.1},
            )
            for i in range(n)
        ]


def _difficulty_temperature(runtime: LLMRuntimeConfig, diff: Difficulty) -> float:
    if diff == Difficulty.EASY:
        return runtime.temperature_easy
    if diff == Difficulty.MEDIUM:
        return runtime.temperature_medium
    return runtime.temperature_hard


def _difficulty_max_tokens(runtime: LLMRuntimeConfig, diff: Difficulty) -> int:
    if diff == Difficulty.EASY:
        return runtime.max_new_tokens_easy
    if diff == Difficulty.MEDIUM:
        return runtime.max_new_tokens_medium
    return runtime.max_new_tokens_hard


class _TextGenerationEngine:
    """Lazy runtime abstraction for vLLM/Transformers."""

    def __init__(self, runtime: LLMRuntimeConfig, model_name: str):
        self.runtime = runtime
        self.model_name = model_name
        self.backend: str | None = None
        self.client: Any = None
        self.tokenizer: Any = None
        self.generator: Any = None
        self.last_error: Exception | None = None

    def _load_vllm(self) -> bool:
        try:
            from vllm import LLM
        except Exception as exc:
            self.last_error = exc
            return False
        try:
            self.client = LLM(
                model=self.model_name,
                tensor_parallel_size=self.runtime.tensor_parallel_size,
                gpu_memory_utilization=self.runtime.gpu_memory_utilization,
                max_model_len=self.runtime.max_model_len,
                trust_remote_code=self.runtime.trust_remote_code,
            )
            self.backend = "vllm"
            return True
        except Exception as exc:
            self.last_error = exc
            return False

    def _load_transformers(self) -> bool:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        except Exception as exc:
            self.last_error = exc
            return False
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=self.runtime.trust_remote_code,
            )
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=self.runtime.trust_remote_code,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            self.generator = pipeline("text-generation", model=model, tokenizer=self.tokenizer)
            self.backend = "transformers"
            return True
        except Exception as exc:
            self.last_error = exc
            return False

    def _build_unavailable_message(self) -> str:
        model_path = Path(self.model_name)
        model_exists = model_path.exists()
        hints = [f"Unable to load model runtime for backend={self.runtime.backend!r}, model={self.model_name!r}."]
        if not model_exists and "/" in self.model_name and not self.model_name.startswith("/"):
            hints.append(
                "Model looks like a repo id. In offline environments, mount local model files and pass "
                "--model-main /path/to/model (or set AIMO3_MODEL_MAIN)."
            )
        if self.runtime.backend.lower() == "vllm":
            hints.append("Ensure vLLM is installed and compatible with the environment/CUDA.")
        if self.last_error is not None:
            hints.append(f"Underlying error: {type(self.last_error).__name__}: {self.last_error}")
        return " ".join(hints)

    def _ensure_loaded(self) -> None:
        if self.backend is not None:
            return
        backend = self.runtime.backend.lower()
        if backend == "heuristic":
            raise InferenceUnavailableError("heuristic runtime cannot be used as text-generation engine")
        if backend in {"auto", "vllm"} and self._load_vllm():
            return
        if backend in {"auto", "transformers"} and self._load_transformers():
            return
        raise InferenceUnavailableError(self._build_unavailable_message())

    def generate(
        self,
        prompt: str,
        *,
        n: int,
        seed: int,
        temperature: float,
        max_new_tokens: int,
        top_p: float,
    ) -> list[str]:
        self._ensure_loaded()
        assert self.backend is not None

        if self.backend == "vllm":
            from vllm import SamplingParams

            params = SamplingParams(
                n=max(1, n),
                temperature=max(0.0, temperature),
                top_p=top_p,
                max_tokens=max_new_tokens,
                seed=seed,
            )
            outputs = self.client.generate([prompt], params)
            texts: list[str] = []
            for item in outputs:
                for out in item.outputs:
                    texts.append(out.text)
            return texts

        if self.backend == "transformers":
            do_sample = temperature > 0
            result = self.generator(
                prompt,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=max(temperature, 1e-5) if do_sample else None,
                top_p=top_p if do_sample else None,
                num_return_sequences=max(1, n),
                pad_token_id=self.tokenizer.eos_token_id if self.tokenizer else None,
            )
            texts: list[str] = []
            for row in result:
                full_text = row["generated_text"]
                if full_text.startswith(prompt):
                    full_text = full_text[len(prompt) :]
                texts.append(full_text.strip())
            return texts

        raise InferenceUnavailableError("Unknown runtime backend state")


def _iter_brace_objects(text: str, max_objects: int = 12) -> list[str]:
    objects: list[str] = []
    start = -1
    depth = 0
    in_str = False
    esc = False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    objects.append(text[start : i + 1])
                    if len(objects) >= max_objects:
                        break
                    start = -1
    return objects


def _extract_first_json_object(text: str) -> dict[str, Any] | None:
    fenced = re.findall(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    candidates = fenced + _iter_brace_objects(text)
    for chunk in candidates:
        try:
            return json.loads(chunk)
        except Exception:
            continue
    return None


def _extract_int(text: str, payload: dict[str, Any] | None) -> int | None:
    if payload:
        for key in ("final_answer", "answer", "final"):
            if key not in payload:
                continue
            value = payload[key]
            if isinstance(value, dict):
                value = value.get("answer")
            if isinstance(value, int):
                return value
            if isinstance(value, str):
                m = re.search(r"-?\d+", value)
                if m:
                    return int(m.group(0))
    boxed = re.findall(r"\\boxed\{(-?\d+)\}", text)
    if boxed:
        return int(boxed[-1])
    final = re.findall(r"FINAL\s*[:=]\s*(-?\d+)", text, flags=re.IGNORECASE)
    if final:
        return int(final[-1])
    ints = re.findall(r"-?\d+", text)
    if ints:
        return int(ints[-1])
    return None


def _extract_code(
    text: str,
    payload: dict[str, Any] | None,
    keys: tuple[str, ...],
) -> str | None:
    if payload:
        for key in keys:
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    fenced = re.findall(r"```python\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced[0].strip()
    return None


def _extract_trace(text: str, payload: dict[str, Any] | None) -> str:
    if payload and isinstance(payload.get("trace"), str):
        return payload["trace"].strip()
    summary = text.strip()
    if len(summary) > 1200:
        summary = summary[:1200]
    return summary


def _extract_confidence_hint(payload: dict[str, Any] | None) -> float | None:
    if not payload:
        return None
    value = payload.get("confidence")
    if isinstance(value, (int, float)):
        return float(min(1.0, max(0.0, value)))
    if isinstance(value, str):
        try:
            v = float(value)
            return float(min(1.0, max(0.0, v)))
        except Exception:
            return None
    return None


def _build_prompt(path: PathType, program: PromptProgram, problem: str, meta: ProblemMetadata) -> str:
    profile = (
        "Program A (tool-heavy): formalize quickly, compute with code, then independent sanity checks."
        if program == PromptProgram.A
        else "Program B (theorem-heavy): derive invariants first, then verify with an independent method."
    )
    modulus_line = (
        f"Extracted modulus hint: {meta.modulus}."
        if meta.modulus is not None
        else "No explicit modulus extracted; infer if present."
    )
    shared = (
        "Return STRICT JSON only.\n"
        "Schema keys allowed: trace, final_answer, python_code, validator_code, confidence.\n"
        "final_answer must be integer.\n"
    )
    path_instruction = {
        PathType.P1_TOOL: (
            "Write solver code in python_code that sets ANSWER = <int>.\n"
            "Write validator_code that checks CANDIDATE_ANSWER and sets IS_VALID = True/False.\n"
            "validator_code must be independent from solver logic when possible.\n"
        ),
        PathType.P2_REASONING: (
            "Provide a concise derivation in trace and final_answer.\n"
            "Include independent verification idea in trace.\n"
        ),
        PathType.P3_BACKSOLVE: (
            "Infer strong constraints first (mod/parity/bounds), then compute final_answer.\n"
            "Include constraint table in trace.\n"
        ),
        PathType.P4_REPAIR: (
            "Critique prior candidates, fix likely error, and provide corrected final_answer.\n"
            "If possible include python_code or validator_code.\n"
        ),
    }[path]
    return (
        "You solve hard olympiad integer problems.\n"
        f"{profile}\n"
        f"{modulus_line}\n"
        f"{shared}{path_instruction}"
        "Final answer must be normalized to [0, 99999] after required reductions.\n"
        f"Problem:\n{problem}\n"
    )


def _build_repair_context(prior_verified: list[VerificationResult], top_k: int = 4) -> str:
    if not prior_verified:
        return "No prior candidates."
    ranked = sorted(prior_verified, key=lambda x: x.score, reverse=True)[:top_k]
    lines = []
    for idx, item in enumerate(ranked, start=1):
        lines.append(
            f"{idx}) answer={item.normalized_answer}, score={item.score:.3f}, "
            f"path={item.candidate.path.value}, hard_ok={item.hard_ok}, notes={'; '.join(item.notes[:2])}"
        )
    return "\n".join(lines)


class CompetitionLLMBackend(BaseLLMBackend):
    """Competition-oriented backend backed by real open-weight models."""

    name = "competition-llm-backend-v2"

    def __init__(self, runtime: LLMRuntimeConfig):
        self.runtime = runtime
        self.main_engine = _TextGenerationEngine(runtime, runtime.model_main)
        self.fast_engine: _TextGenerationEngine | None = None
        if runtime.model_fast and runtime.model_fast != runtime.model_main:
            self.fast_engine = _TextGenerationEngine(runtime, runtime.model_fast)

    def validate_runtime(self) -> None:
        self.main_engine._ensure_loaded()

    def _engine_for_path(self, path: PathType) -> _TextGenerationEngine:
        if path in {PathType.P3_BACKSOLVE, PathType.P4_REPAIR} and self.fast_engine is not None:
            return self.fast_engine
        return self.main_engine

    def _generate_candidates(
        self,
        *,
        path: PathType,
        problem: str,
        meta: ProblemMetadata,
        route: RouteDecision,
        n: int,
        seed: int,
        program: PromptProgram,
        prior_verified: list[VerificationResult] | None = None,
    ) -> list[Candidate]:
        prompt = _build_prompt(path, program, problem, meta)
        if path == PathType.P4_REPAIR:
            repair_context = _build_repair_context(prior_verified or [])
            prompt = f"{prompt}\nPrior candidate summary:\n{repair_context}\n"

        engine = self._engine_for_path(path)
        temperature = _difficulty_temperature(self.runtime, route.difficulty)
        if path == PathType.P4_REPAIR:
            temperature = max(0.1, temperature * 0.75)
        max_new_tokens = _difficulty_max_tokens(self.runtime, route.difficulty)
        texts = engine.generate(
            prompt=prompt,
            n=max(1, n),
            seed=seed,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            top_p=self.runtime.top_p,
        )

        out: list[Candidate] = []
        for i, text in enumerate(texts):
            payload = _extract_first_json_object(text)
            answer = _extract_int(text, payload)
            solver_code = _extract_code(text, payload, ("python_code", "code", "solver_code"))
            validator_code = _extract_code(text, payload, ("validator_code", "check_code", "verification_code"))
            trace = _extract_trace(text, payload)
            conf = _extract_confidence_hint(payload)

            if path == PathType.P1_TOOL and not solver_code and answer is not None:
                solver_code = f"ANSWER = {answer}"

            cand = Candidate(
                path=path,
                answer=answer if path not in {PathType.P1_TOOL} else None,
                python_code=solver_code if path in {PathType.P1_TOOL, PathType.P4_REPAIR} else None,
                trace=trace,
                program=program,
                metadata={
                    "backend": self.name,
                    "sample_index": i,
                    "seed": seed,
                    "raw_output": text[:4000],
                },
            )
            if validator_code:
                cand.metadata["validator_code"] = validator_code
            if conf is not None:
                cand.metadata["confidence_hint"] = conf
            out.append(cand)
        return out

    def generate_tool_candidates(
        self,
        problem: str,
        meta: ProblemMetadata,
        route: RouteDecision,
        n: int,
        seed: int,
        program: PromptProgram,
    ) -> list[Candidate]:
        return self._generate_candidates(
            path=PathType.P1_TOOL,
            problem=problem,
            meta=meta,
            route=route,
            n=n,
            seed=seed,
            program=program,
        )

    def generate_reasoning_candidates(
        self,
        problem: str,
        meta: ProblemMetadata,
        route: RouteDecision,
        n: int,
        seed: int,
        program: PromptProgram,
    ) -> list[Candidate]:
        return self._generate_candidates(
            path=PathType.P2_REASONING,
            problem=problem,
            meta=meta,
            route=route,
            n=n,
            seed=seed,
            program=program,
        )

    def generate_backsolve_candidates(
        self,
        problem: str,
        meta: ProblemMetadata,
        route: RouteDecision,
        n: int,
        seed: int,
        program: PromptProgram,
    ) -> list[Candidate]:
        return self._generate_candidates(
            path=PathType.P3_BACKSOLVE,
            problem=problem,
            meta=meta,
            route=route,
            n=n,
            seed=seed,
            program=program,
        )

    def generate_repair_candidates(
        self,
        problem: str,
        meta: ProblemMetadata,
        route: RouteDecision,
        n: int,
        seed: int,
        program: PromptProgram,
        prior_verified: list[VerificationResult],
    ) -> list[Candidate]:
        return self._generate_candidates(
            path=PathType.P4_REPAIR,
            problem=problem,
            meta=meta,
            route=route,
            n=n,
            seed=seed,
            program=program,
            prior_verified=prior_verified,
        )
