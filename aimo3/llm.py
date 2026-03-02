from __future__ import annotations

import hashlib
from dataclasses import dataclass

from aimo3.models import Candidate, PathType, ProblemMetadata, PromptProgram, RouteDecision


@dataclass
class NeuralJudge:
    """Small heuristic judge placeholder.

    This class is intentionally lightweight and deterministic so that the
    project is runnable without model checkpoints. It can be replaced by an
    actual trained judge model.
    """

    name: str = "heuristic-judge-v1"

    def score(self, problem: str, trace: str, answer: int, artifacts: dict) -> float:
        base = 0.45
        if artifacts.get("tool_ok"):
            base += 0.2
        if "invariant" in trace.lower() or "constraint" in trace.lower():
            base += 0.1
        if "timeout" in artifacts.get("error", "").lower():
            base -= 0.25
        if answer < 0:
            base -= 0.4
        return float(min(1.0, max(0.0, base)))


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
    """Deterministic local backend that mimics multi-path candidate generation."""

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
            nums = meta.numbers[:12]
            nums_literal = repr(nums if nums else [seed % 9973, (seed // 7) % 7919])
            modulus = meta.modulus or 100000
            code = (
                "from fractions import Fraction\n"
                f"nums = {nums_literal}\n"
                f"modulus = {modulus}\n"
                "acc = 0\n"
                "for idx, n in enumerate(nums, start=1):\n"
                "    acc += (n * n + 17 * idx)\n"
                f"acc += {answer}\n"
                "ANSWER = acc % modulus\n"
            )
            trace = (
                "Tool-integrated derivation: construct arithmetic invariant from extracted constants, "
                "evaluate with Python, then reduce by modulus."
            )
            out.append(
                Candidate(
                    path=PathType.P1_TOOL,
                    answer=None,
                    python_code=code,
                    trace=trace,
                    program=program,
                    metadata={"backend": self.name, "attempt": i, "seed": seed + i},
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
        out: list[Candidate] = []
        for i in range(n):
            answer = self._hash_answer(problem, f"reason-{program.value}-{i}", seed + 31 * i, meta.modulus)
            trace = (
                "Reasoning-only path: identify dominant invariant, derive reduced expression, "
                "then normalize final value under extracted constraints."
            )
            out.append(
                Candidate(
                    path=PathType.P2_REASONING,
                    answer=answer,
                    trace=trace,
                    program=program,
                    metadata={"backend": self.name, "attempt": i, "seed": seed + 31 * i},
                )
            )
        return out

    def generate_backsolve_candidates(
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
            answer = self._hash_answer(problem, f"backsolve-{program.value}-{i}", seed + 97 * i, meta.modulus)
            trace = (
                "Backsolve path: infer constraints on answer class (parity/modular residues), "
                "search candidate set, and return best consistent residue."
            )
            out.append(
                Candidate(
                    path=PathType.P3_BACKSOLVE,
                    answer=answer,
                    trace=trace,
                    program=program,
                    metadata={"backend": self.name, "attempt": i, "seed": seed + 97 * i},
                )
            )
        return out
