from __future__ import annotations

import json
import os
from pathlib import Path

from aimo3.budget import allocate_budget
from aimo3.config import SolverConfig
from aimo3.debug import DebugTracer
from aimo3.generator import CandidateGenerator
from aimo3.hard_mode import HardModeEngine
from aimo3.memory import MemoryRetriever
from aimo3.llm import (
    BaseLLMBackend,
    CompetitionLLMBackend,
    HeuristicLLMBackend,
    InferenceUnavailableError,
    NeuralJudge,
)
from aimo3.models import Candidate, PathType, PromptProgram, SolveResult, VerificationResult
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
        self.memory = MemoryRetriever(self.config.reference_path) if self.config.allow_reference_lookup else None
        self.run_seed = int.from_bytes(os.urandom(4), "big")
        self.tracer = DebugTracer(
            enabled=self.config.debug_enabled,
            max_chars=self.config.debug_max_chars,
            file_path=self.config.debug_file_path,
        )
        self.config.run_log_dir.mkdir(parents=True, exist_ok=True)

    def _pick_prompt_program(self, run_seed: int) -> PromptProgram:
        return PromptProgram.A if run_seed % 2 == 0 else PromptProgram.B

    def _fallback_answer(self, statement_hash: str) -> int:
        return int(statement_hash[:12], 16) % self.config.global_modulus_fallback

    def _write_log(self, result: SolveResult) -> None:
        path = Path(self.config.run_log_dir) / f"{result.pid}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(result.to_log_dict(), f, ensure_ascii=False, indent=2)
        self._log("result_log_written", path=str(path))

    def _log(self, event: str, **payload) -> None:
        self.tracer.log(event, **payload)

    def _candidate_summary(self, cand: Candidate) -> dict:
        out = {
            "path": cand.path.value,
            "program": cand.program.value,
            "answer": cand.answer,
            "has_python_code": bool(cand.python_code),
            "trace_preview": cand.trace[:300],
            "metadata_keys": sorted(cand.metadata.keys()),
        }
        if self.config.debug_include_raw_output:
            if cand.python_code:
                out["python_code"] = cand.python_code
            raw_output = cand.metadata.get("raw_output")
            if isinstance(raw_output, str):
                out["raw_output"] = raw_output
        return out

    def _verified_summary(self, item: VerificationResult) -> dict:
        out = {
            "path": item.candidate.path.value,
            "normalized_answer": item.normalized_answer,
            "score": item.score,
            "hard_ok": item.hard_ok,
            "symbolic_ok": item.symbolic_ok,
            "random_ok": item.random_ok,
            "judge_prob": item.judge_prob,
            "vote_share": item.self_consistency_vote_share,
            "path_diversity": item.path_diversity,
            "notes": item.notes,
            "has_validator": "validator_code" in item.candidate.metadata,
        }
        if self.config.debug_include_raw_output:
            out["artifacts"] = item.artifacts
            out["candidate"] = self._candidate_summary(item.candidate)
        return out

    def _memory_candidate(self, meta) -> Candidate | None:
        if self.memory is None:
            return None
        match = self.memory.lookup(meta, threshold=self.config.reference_similarity_threshold)
        if match is None:
            return None
        confidence_hint = 0.98 if match.similarity >= 0.999 else 0.60
        return Candidate(
            path=PathType.P5_MEMORY,
            answer=match.answer,
            trace=f"Memory retrieval match id={match.source_id} similarity={match.similarity:.4f}",
            program=PromptProgram.A,
            metadata={
                "memory_match_id": match.source_id,
                "memory_similarity": match.similarity,
                "confidence_hint": confidence_hint,
                "skip_modulus_normalization": True,
            },
        )

    def solve_one(self, pid: str, latex: str, run_seed: int | None = None) -> SolveResult:
        seed = self.run_seed if run_seed is None else run_seed
        self._log("solve_start", pid=pid, run_seed=seed, text_chars=len(latex))
        meta = parse_problem(pid, latex)
        self._log(
            "parse_done",
            pid=pid,
            statement_hash=meta.statement_hash,
            modulus=meta.modulus,
            numbers_preview=meta.numbers[:20],
            equation_count=len(meta.extracted_equations),
            domain_hints=meta.detected_domain_hints,
        )
        route = route_problem(meta, self.config)
        self._log(
            "route_done",
            pid=pid,
            domain=route.domain.value,
            difficulty=route.difficulty.value,
            difficulty_score=route.difficulty_score,
            requires_tool=route.requires_tool,
            use_backsolve=route.use_backsolve,
            allow_hard_mode=route.allow_hard_mode,
            n_tool_attempts=route.n_tool_attempts,
            n_cot_attempts=route.n_cot_attempts,
            n_backsolve_attempts=route.n_backsolve_attempts,
        )
        budget = allocate_budget(route, self.config)
        self._log(
            "budget_allocated",
            pid=pid,
            time_limit_s=budget.time_limit_s,
            max_attempts=budget.max_attempts,
            max_tool_runs=budget.max_tool_runs,
            max_output_tokens=budget.max_output_tokens,
        )
        program = self._pick_prompt_program(seed)
        self._log("program_selected", pid=pid, program=program.value)

        verified: list[VerificationResult] = []
        memory_cand = self._memory_candidate(meta)
        if memory_cand is not None:
            self._log("memory_candidate", pid=pid, candidate=self._candidate_summary(memory_cand))
            verified.extend(self.verifier.verify_batch([memory_cand], latex, meta))
            self._log(
                "memory_verified",
                pid=pid,
                verified=[self._verified_summary(v) for v in verified],
            )
            selected, reason = self.verifier.select_final(verified)
            if selected is not None and selected.normalized_answer is not None:
                sim = float(memory_cand.metadata.get("memory_similarity", 0.0))
                result = SolveResult(
                    pid=pid,
                    answer=int(selected.normalized_answer),
                    meta=meta,
                    route=route,
                    candidates=verified,
                    selected=selected,
                    reason=f"{reason}|memory_match_{sim:.4f}",
                )
                self._log(
                    "solve_end_memory_shortcut",
                    pid=pid,
                    answer=result.answer,
                    reason=result.reason,
                    selected=self._verified_summary(selected),
                )
                self._write_log(result)
                return result

        if route.try_symbolic_first:
            sym_candidates = symbolic_first_pass(meta)
            self._log(
                "symbolic_candidates_generated",
                pid=pid,
                count=len(sym_candidates),
                candidates=[self._candidate_summary(c) for c in sym_candidates],
            )
            budget.use_attempt(len(sym_candidates))
            verified.extend(self.verifier.verify_batch(sym_candidates, latex, meta))
            self._log(
                "symbolic_candidates_verified",
                pid=pid,
                count=len(sym_candidates),
                attempts_used=budget.attempts_used,
                verified=[self._verified_summary(v) for v in verified[-len(sym_candidates) :]] if sym_candidates else [],
            )

        stale_rounds = 0
        repair_rounds = 0
        round_index = 0
        while budget.can_attempt() and budget.remaining_time() > 0:
            self._log(
                "loop_begin",
                pid=pid,
                round_index=round_index,
                attempts_used=budget.attempts_used,
                tool_runs_used=budget.tool_runs_used,
                remaining_time_s=budget.remaining_time(),
                verified_count=len(verified),
            )
            if self.verifier.confident_enough(verified):
                self._log("loop_break_confident", pid=pid, round_index=round_index)
                break
            if len(verified) >= self.config.max_candidates:
                self._log("loop_break_max_candidates", pid=pid, round_index=round_index, max_candidates=self.config.max_candidates)
                break

            new_candidates = self.generator.generate_multimodal(
                problem=latex,
                meta=meta,
                route=route,
                budget=budget,
                run_seed=seed + budget.attempts_used,
                program=program,
                round_index=round_index,
            )
            if not new_candidates:
                self._log("loop_break_no_candidates", pid=pid, round_index=round_index)
                break
            self._log(
                "candidates_generated",
                pid=pid,
                round_index=round_index,
                count=len(new_candidates),
                candidates=[self._candidate_summary(c) for c in new_candidates],
            )
            budget.use_attempt(len(new_candidates))
            new_verified = self.verifier.verify_batch(new_candidates, latex, meta)
            verified.extend(new_verified)
            self._log(
                "candidates_verified",
                pid=pid,
                round_index=round_index,
                count=len(new_verified),
                attempts_used=budget.attempts_used,
                verified=[self._verified_summary(v) for v in new_verified],
            )

            valid_now = sum(v.hard_ok for v in new_verified)
            if valid_now == 0:
                stale_rounds += 1
            else:
                stale_rounds = 0
            self._log(
                "round_quality",
                pid=pid,
                round_index=round_index,
                valid_now=valid_now,
                stale_rounds=stale_rounds,
                repair_rounds=repair_rounds,
            )

            if (
                repair_rounds < self.config.max_repair_rounds
                and valid_now > 0
                and budget.can_attempt()
                and not self.verifier.confident_enough(verified)
            ):
                repair_candidates = self.generator.generate_repair(
                    problem=latex,
                    meta=meta,
                    route=route,
                    budget=budget,
                    run_seed=seed + 503 + repair_rounds * 17,
                    program=program,
                    prior_verified=verified,
                    top_k=self.config.repair_top_k,
                )
                if repair_candidates:
                    self._log(
                        "repair_generated",
                        pid=pid,
                        round_index=round_index,
                        repair_round=repair_rounds,
                        count=len(repair_candidates),
                        candidates=[self._candidate_summary(c) for c in repair_candidates],
                    )
                    budget.use_attempt(len(repair_candidates))
                    repair_verified = self.verifier.verify_batch(repair_candidates, latex, meta)
                    verified.extend(repair_verified)
                    self._log(
                        "repair_verified",
                        pid=pid,
                        round_index=round_index,
                        repair_round=repair_rounds,
                        verified=[self._verified_summary(v) for v in repair_verified],
                    )
                    repair_rounds += 1

            if stale_rounds >= 2 and route.allow_hard_mode:
                hard_candidates = self.hard_mode.run(
                    problem=latex,
                    meta=meta,
                    route=route,
                    budget=budget,
                    run_seed=seed + 1001,
                    program=program,
                    prior_verified=verified,
                )
                self._log(
                    "hard_mode_generated",
                    pid=pid,
                    round_index=round_index,
                    count=len(hard_candidates),
                    candidates=[self._candidate_summary(c) for c in hard_candidates],
                )
                hard_verified = self.verifier.verify_batch(hard_candidates, latex, meta)
                verified.extend(hard_verified)
                self._log(
                    "hard_mode_verified",
                    pid=pid,
                    round_index=round_index,
                    verified=[self._verified_summary(v) for v in hard_verified],
                )
                stale_rounds = 0
                program = PromptProgram.B if program == PromptProgram.A else PromptProgram.A
                self._log("program_switched", pid=pid, round_index=round_index, program=program.value)
            round_index += 1

        selected, reason = self.verifier.select_final(verified)
        self._log(
            "selection_done",
            pid=pid,
            selected=self._verified_summary(selected) if selected else None,
            total_candidates=len(verified),
            reason=reason,
        )
        if selected is not None and selected.normalized_answer is not None:
            answer = selected.normalized_answer
        else:
            if self.config.enforce_real_backend and not self.config.allow_demo_fallback:
                self._log("solve_error_no_valid_candidate", pid=pid, strict_mode=True)
                raise RuntimeError("No valid answer candidate produced under strict competition mode.")
            answer = self._fallback_answer(meta.statement_hash)
            reason = f"{reason}|fallback_hash"
            self._log("fallback_used", pid=pid, fallback_answer=answer, reason=reason)

        result = SolveResult(
            pid=pid,
            answer=int(answer),
            meta=meta,
            route=route,
            candidates=verified,
            selected=selected,
            reason=reason,
        )
        self._log(
            "solve_end",
            pid=pid,
            answer=result.answer,
            reason=result.reason,
            attempts_used=budget.attempts_used,
            tool_runs_used=budget.tool_runs_used,
            elapsed_s=(budget.time_limit_s - budget.remaining_time()),
        )
        self._write_log(result)
        return result
