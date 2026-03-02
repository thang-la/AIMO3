from __future__ import annotations

from aimo3.budget import BudgetManager
from aimo3.llm import BaseLLMBackend
from aimo3.models import Candidate, ProblemMetadata, PromptProgram, RouteDecision


class CandidateGenerator:
    def __init__(self, llm_main: BaseLLMBackend):
        self.llm_main = llm_main

    def generate_multimodal(
        self,
        problem: str,
        meta: ProblemMetadata,
        route: RouteDecision,
        budget: BudgetManager,
        run_seed: int,
        program: PromptProgram,
    ) -> list[Candidate]:
        out: list[Candidate] = []

        if route.requires_tool and budget.can_run_tool():
            tool_candidates = self.llm_main.generate_tool_candidates(
                problem=problem,
                meta=meta,
                route=route,
                n=route.n_tool_attempts,
                seed=run_seed,
                program=program,
            )
            out.extend(tool_candidates)
            budget.use_tool_run()

        reasoning_candidates = self.llm_main.generate_reasoning_candidates(
            problem=problem,
            meta=meta,
            route=route,
            n=route.n_cot_attempts,
            seed=run_seed + 19,
            program=program,
        )
        out.extend(reasoning_candidates)

        if route.use_backsolve:
            out.extend(
                self.llm_main.generate_backsolve_candidates(
                    problem=problem,
                    meta=meta,
                    route=route,
                    n=route.n_backsolve_attempts,
                    seed=run_seed + 41,
                    program=program,
                )
            )
        return out
