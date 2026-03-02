from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

from aimo3.config import LLMRuntimeConfig
from aimo3.models import Candidate, Difficulty, PathType, ProblemMetadata, PromptProgram, RouteDecision


@dataclass
class NeuralJudge:
    """Heuristic judge that can be swapped with a trained model."""

    name: str = "heuristic-judge-v1"

    def score(self, problem: str, trace: str, answer: int, artifacts: dict) -> float:
        base = 0.5
        if artifacts.get("tool_ok"):
            base += 0.25
        if "independent check" in trace.lower() or "verify" in trace.lower():
            base += 0.1
        if artifacts.get("sandbox_timeout"):
            base -= 0.2
        if answer < 0:
            base -= 0.5
        return float(min(1.0, max(0.0, base)))


class InferenceUnavailableError(RuntimeError):
    """Raised when no configured runtime backend can be loaded."""


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


class HeuristicLLMBackend(BaseLLMBackend):
    """Deterministic fallback backend used only when explicitly allowed."""

    name = "heuristic-backend-v1"

    @staticmethod
    def _hash_answer(problem: str, tag: str, seed: int, modulus: int | None) -> int:
        key = f"{problem}|{tag}|{seed}"
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        value = int(digest[:16], 16)
        if modulus is not None and modulus > 0:
            return value % modulus
        return value % 100000

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
            code = (
                f"modulus = {meta.modulus or 100000}\n"
                f"ANSWER = {answer} % modulus\n"
            )
            out.append(
                Candidate(
                    path=PathType.P1_TOOL,
                    answer=None,
                    python_code=code,
                    trace="Fallback heuristic tool path.",
                    program=program,
                    metadata={"backend": self.name, "attempt": i},
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
                trace="Fallback heuristic reasoning path.",
                program=program,
                metadata={"backend": self.name, "attempt": i},
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
                trace="Fallback heuristic backsolve path.",
                program=program,
                metadata={"backend": self.name, "attempt": i},
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

    def _load_vllm(self) -> bool:
        try:
            from vllm import LLM
        except Exception:
            return False
        self.client = LLM(
            model=self.model_name,
            tensor_parallel_size=self.runtime.tensor_parallel_size,
            gpu_memory_utilization=self.runtime.gpu_memory_utilization,
            max_model_len=self.runtime.max_model_len,
            trust_remote_code=self.runtime.trust_remote_code,
        )
        self.backend = "vllm"
        return True

    def _load_transformers(self) -> bool:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        except Exception:
            return False
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
        self.generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=self.tokenizer,
        )
        self.backend = "transformers"
        return True

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
        raise InferenceUnavailableError(
            f"Unable to load model runtime for backend={self.runtime.backend!r}, model={self.model_name!r}"
        )

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
            do_sample = temperature > 0.0
            result = self.generator(
                prompt,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=max(temperature, 1e-5) if do_sample else None,
                top_p=top_p if do_sample else None,
                num_return_sequences=max(1, n),
                pad_token_id=self.tokenizer.eos_token_id if self.tokenizer else None,
            )
            texts = []
            for row in result:
                full_text = row["generated_text"]
                texts.append(full_text[len(prompt) :].strip() if full_text.startswith(prompt) else full_text.strip())
            return texts

        raise InferenceUnavailableError("Unknown backend state")


def _extract_first_json_object(text: str) -> dict[str, Any] | None:
    fenced = re.findall(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    candidates = fenced + re.findall(r"(\{.*\})", text, flags=re.DOTALL)
    for chunk in candidates:
        try:
            return json.loads(chunk)
        except Exception:
            continue
    return None


def _extract_python_code(text: str, payload: dict[str, Any] | None) -> str | None:
    if payload:
        code = payload.get("python_code") or payload.get("code")
        if isinstance(code, str) and code.strip():
            return code.strip()
    fenced = re.findall(r"```python\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced[0].strip()
    return None


def _extract_int(text: str, payload: dict[str, Any] | None) -> int | None:
    if payload:
        for key in ("final_answer", "answer", "final"):
            if key in payload:
                value = payload[key]
                if isinstance(value, dict):
                    if "answer" in value:
                        value = value["answer"]
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
    all_ints = re.findall(r"-?\d+", text)
    if all_ints:
        return int(all_ints[-1])
    return None


def _extract_trace(text: str, payload: dict[str, Any] | None) -> str:
    if payload and isinstance(payload.get("trace"), str):
        return payload["trace"].strip()
    summary = text.strip()
    if len(summary) > 900:
        summary = summary[:900]
    return summary


def _build_prompt(
    path: PathType,
    program: PromptProgram,
    problem: str,
    meta: ProblemMetadata,
) -> str:
    profile = (
        "Program A: tool-first, conservative, verify each step."
        if program == PromptProgram.A
        else "Program B: theorem-first then independent computational confirmation."
    )
    modulus_line = f"Extracted modulus hint: {meta.modulus}." if meta.modulus is not None else "No explicit modulus extracted."
    instructions = {
        PathType.P1_TOOL: (
            "Produce Python code to compute the final integer answer.\n"
            "Return JSON with keys: trace, python_code, final_answer.\n"
            "Code must set ANSWER = <int>.\n"
        ),
        PathType.P2_REASONING: (
            "Solve by mathematical reasoning.\n"
            "Return JSON with keys: trace, final_answer.\n"
            "Also include FINAL: <int> at the end.\n"
        ),
        PathType.P3_BACKSOLVE: (
            "Infer constraints on the answer (modular residues, parity, bounds), then determine final answer.\n"
            "Return JSON with keys: trace, final_answer.\n"
        ),
    }[path]

    return (
        "You are solving an olympiad-style integer answer problem.\n"
        f"{profile}\n"
        f"{modulus_line}\n"
        "Final answer must be an integer in [0, 99999] after required reductions.\n"
        f"{instructions}\n"
        f"Problem:\n{problem}\n"
    )


class CompetitionLLMBackend(BaseLLMBackend):
    """Competition-oriented backend backed by real open-weight models."""

    name = "competition-llm-backend"

    def __init__(self, runtime: LLMRuntimeConfig):
        self.runtime = runtime
        self.main_engine = _TextGenerationEngine(runtime, runtime.model_main)

    def validate_runtime(self) -> None:
        self.main_engine._ensure_loaded()

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
    ) -> list[Candidate]:
        prompt = _build_prompt(path, program, problem, meta)
        temperature = _difficulty_temperature(self.runtime, route.difficulty)
        max_new_tokens = _difficulty_max_tokens(self.runtime, route.difficulty)
        texts = self.main_engine.generate(
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
            py_code = _extract_python_code(text, payload)
            trace = _extract_trace(text, payload)
            if path == PathType.P1_TOOL and not py_code and answer is not None:
                py_code = f"ANSWER = {answer}"
            out.append(
                Candidate(
                    path=path,
                    answer=answer if path != PathType.P1_TOOL else None,
                    python_code=py_code if path == PathType.P1_TOOL else None,
                    trace=trace,
                    program=program,
                    metadata={
                        "backend": self.name,
                        "sample_index": i,
                        "seed": seed,
                        "raw_output": text[:4000],
                    },
                )
            )
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
