from __future__ import annotations

import json
import os
from pathlib import Path

from aimo3.budget import allocate_budget
from aimo3.config import SolverConfig
from aimo3.generator import CandidateGenerator
from aimo3.hard_mode import HardModeEngine
from aimo3.llm import (
    BaseLLMBackend,
    CompetitionLLMBackend,
    HeuristicLLMBackend,
    InferenceUnavailableError,
    NeuralJudge,
)
from aimo3.models import PromptProgram, SolveResult, VerificationResult
from aimo3.parsing import parse_problem
from aimo3.router import route_problem
from aimo3.symbolic import symbolic_first_pass
from aimo3.verifier import Verifier


def _default_backend(config: SolverConfig) -> BaseLLMBackend:
    mode = config.llm.backend.lower()
    if mode == "heuristic":
        if config.enforce_real_backend and not config.allow_demo_fallback:
            raise InferenceUnavailableError(
                "Heuristic backend is disabled. Set allow_demo_fallback=True to use it explicitly."
            )
        return HeuristicLLMBackend()

    backend = CompetitionLLMBackend(config.llm)
    if config.enforce_real_backend:
        backend.validate_runtime()
    return backend


class AIMO3Solver:
    def __init__(
        self,
        config: SolverConfig | None = None,
        llm_main: BaseLLMBackend | None = None,
        llm_fast: BaseLLMBackend | None = None,
        judge: NeuralJudge | None = None,
    ):
        self.config = config or SolverConfig()
        self.llm_main = llm_main or _default_backend(self.config)
        self.llm_fast = llm_fast or self.llm_main
        self.judge = judge or NeuralJudge()
        self.generator = CandidateGenerator(self.llm_main)
        self.hard_mode = HardModeEngine(self.llm_fast)
        self.verifier = Verifier(self.config, self.judge)
        self.run_seed = int.from_bytes(os.urandom(4), "big")
        self.config.run_log_dir.mkdir(parents=True, exist_ok=True)

    def _pick_prompt_program(self, run_seed: int) -> PromptProgram:
        return PromptProgram.A if run_seed % 2 == 0 else PromptProgram.B

    def _fallback_answer(self, statement_hash: str) -> int:
        return int(statement_hash[:12], 16) % self.config.global_modulus_fallback

    def _write_log(self, result: SolveResult) -> None:
        path = Path(self.config.run_log_dir) / f"{result.pid}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(result.to_log_dict(), f, ensure_ascii=False, indent=2)

    def solve_one(self, pid: str, latex: str, run_seed: int | None = None) -> SolveResult:
        seed = self.run_seed if run_seed is None else run_seed
        meta = parse_problem(pid, latex)
        route = route_problem(meta, self.config)
        budget = allocate_budget(route, self.config)
        program = self._pick_prompt_program(seed)

        verified: list[VerificationResult] = []
        if route.try_symbolic_first:
            sym_candidates = symbolic_first_pass(meta)
            budget.use_attempt(len(sym_candidates))
            verified.extend(self.verifier.verify_batch(sym_candidates, latex, meta))

        stale_rounds = 0
        while budget.can_attempt() and budget.remaining_time() > 0:
            if self.verifier.confident_enough(verified):
                break
            if len(verified) >= self.config.max_candidates:
                break

            new_candidates = self.generator.generate_multimodal(
                problem=latex,
                meta=meta,
                route=route,
                budget=budget,
                run_seed=seed + budget.attempts_used,
                program=program,
            )
            if not new_candidates:
                break
            budget.use_attempt(len(new_candidates))
            new_verified = self.verifier.verify_batch(new_candidates, latex, meta)
            verified.extend(new_verified)

            valid_now = sum(v.hard_ok for v in new_verified)
            if valid_now == 0:
                stale_rounds += 1
            else:
                stale_rounds = 0

            if stale_rounds >= 2 and route.allow_hard_mode:
                hard_candidates = self.hard_mode.run(
                    problem=latex,
                    meta=meta,
                    route=route,
                    budget=budget,
                    run_seed=seed + 1001,
                    program=program,
                )
                verified.extend(self.verifier.verify_batch(hard_candidates, latex, meta))
                stale_rounds = 0
                program = PromptProgram.B if program == PromptProgram.A else PromptProgram.A

        selected, reason = self.verifier.select_final(verified)
        if selected is not None and selected.normalized_answer is not None:
            answer = selected.normalized_answer
        else:
            if self.config.enforce_real_backend and not self.config.allow_demo_fallback:
                raise RuntimeError("No valid answer candidate produced under strict competition mode.")
            answer = self._fallback_answer(meta.statement_hash)
            reason = f"{reason}|fallback_hash"

        result = SolveResult(
            pid=pid,
            answer=int(answer),
            meta=meta,
            route=route,
            candidates=verified,
            selected=selected,
            reason=reason,
        )
        self._write_log(result)
        return result
