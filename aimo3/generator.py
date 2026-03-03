from __future__ import annotations

from aimo3.budget import BudgetManager
from aimo3.llm import BaseLLMBackend
from aimo3.models import Candidate, ProblemMetadata, PromptProgram, RouteDecision, VerificationResult


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
        round_index: int = 0,
    ) -> list[Candidate]:
        out: list[Candidate] = []
        active_program = program if round_index % 2 == 0 else (PromptProgram.B if program == PromptProgram.A else PromptProgram.A)

        if route.requires_tool and budget.can_run_tool():
            tool_candidates = self.llm_main.generate_tool_candidates(
                problem=problem,
                meta=meta,
                route=route,
                n=route.n_tool_attempts,
                seed=run_seed,
                program=active_program,
            )
            out.extend(tool_candidates)
            budget.use_tool_run()

        reasoning_candidates = self.llm_main.generate_reasoning_candidates(
            problem=problem,
            meta=meta,
            route=route,
            n=route.n_cot_attempts,
            seed=run_seed + 19,
            program=active_program,
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
                    program=active_program,
                )
            )
        return out

    def generate_repair(
        self,
        problem: str,
        meta: ProblemMetadata,
        route: RouteDecision,
        budget: BudgetManager,
        run_seed: int,
        program: PromptProgram,
        prior_verified: list[VerificationResult],
        top_k: int,
    ) -> list[Candidate]:
        if not prior_verified:
            return []
        if not budget.can_attempt():
            return []

        ranked = sorted(prior_verified, key=lambda x: x.score, reverse=True)
        focus = ranked[: max(1, top_k)]
        repair_candidates = self.llm_main.generate_repair_candidates(
            problem=problem,
            meta=meta,
            route=route,
            n=max(1, min(3, len(focus))),
            seed=run_seed,
            program=program,
            prior_verified=focus,
        )
        if any(c.python_code for c in repair_candidates) and budget.can_run_tool():
            budget.use_tool_run()
        return repair_candidates
