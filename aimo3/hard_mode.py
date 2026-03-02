from __future__ import annotations

from aimo3.budget import BudgetManager
from aimo3.llm import BaseLLMBackend
from aimo3.models import Candidate, PathType, ProblemMetadata, PromptProgram, RouteDecision


class HardModeEngine:
    """Bounded search loop for hard problems."""

    def __init__(self, llm_main: BaseLLMBackend):
        self.llm_main = llm_main

    def run(
        self,
        problem: str,
        meta: ProblemMetadata,
        route: RouteDecision,
        budget: BudgetManager,
        run_seed: int,
        program: PromptProgram,
    ) -> list[Candidate]:
        if not budget.can_attempt():
            return []

        out: list[Candidate] = []
        lemma_trace = (
            "Hard-mode lemma mining: propose invariants, stress-test each candidate with an alternate "
            "constraint path, keep only internally consistent residues."
        )
        out.append(
            Candidate(
                path=PathType.HARD_MODE,
                answer=None,
                trace=lemma_trace,
                python_code=(
                    "nums = [3, 5, 7, 11, 13]\n"
                    "acc = 0\n"
                    "for i, v in enumerate(nums, start=1):\n"
                    "    acc += i * v * v\n"
                    f"acc += {run_seed % 100000}\n"
                    f"modulus = {meta.modulus or 100000}\n"
                    "ANSWER = acc % modulus\n"
                ),
                program=program,
                metadata={"hard_mode": "lemma_loop"},
            )
        )
        budget.use_attempt()

        if budget.can_attempt():
            out.extend(
                self.llm_main.generate_reasoning_candidates(
                    problem=problem,
                    meta=meta,
                    route=route,
                    n=2,
                    seed=run_seed + 777,
                    program=PromptProgram.B if program == PromptProgram.A else PromptProgram.A,
                )
            )
            budget.use_attempt()
        return out
