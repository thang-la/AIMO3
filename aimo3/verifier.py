from __future__ import annotations

from collections import Counter

from aimo3.config import SolverConfig, VerificationWeights
from aimo3.llm import NeuralJudge
from aimo3.models import Candidate, ProblemMetadata, VerificationResult
from aimo3.sandbox import run_python_sandbox


class Verifier:
    def __init__(self, config: SolverConfig, judge: NeuralJudge):
        self.config = config
        self.weights = VerificationWeights()
        self.judge = judge

    def _normalize_answer(self, ans: int | None, meta: ProblemMetadata) -> int | None:
        if ans is None:
            return None
        if meta.modulus is not None and meta.modulus > 0:
            ans = ans % meta.modulus
        ans = ans % self.config.global_modulus_fallback
        return int(ans)

    def _check_range(self, ans: int | None) -> bool:
        if ans is None:
            return False
        return 0 <= ans <= self.config.answer_upper_bound

    def verify_candidate(self, candidate: Candidate, problem: str, meta: ProblemMetadata) -> VerificationResult:
        vr = VerificationResult(candidate=candidate)
        answer = candidate.answer
        artifacts: dict[str, object] = {}

        if answer is None and candidate.python_code:
            sandbox = run_python_sandbox(
                candidate.python_code,
                timeout_s=self.config.sandbox_timeout_s,
                memory_mb=self.config.sandbox_memory_mb,
            )
            artifacts["sandbox"] = {
                "success": sandbox.success,
                "timeout": sandbox.timeout,
                "stdout": sandbox.stdout,
                "stderr": sandbox.stderr,
                "error": sandbox.error,
            }
            if sandbox.success:
                answer = sandbox.answer
                artifacts["tool_ok"] = True
            else:
                artifacts["tool_ok"] = False
                vr.sandbox_error = True
                vr.notes.append("sandbox execution failed")

        normalized = self._normalize_answer(answer, meta)
        vr.normalized_answer = normalized
        vr.hard_ok = self._check_range(normalized)
        if not vr.hard_ok:
            vr.notes.append("hard constraints failed (range/format)")

        # Lightweight symbolic consistency heuristic:
        # if equations exist and candidate exists, require at least integer answer.
        vr.symbolic_ok = vr.hard_ok and (len(meta.extracted_equations) == 0 or normalized is not None)

        # Randomized check approximation: candidates from tool path are trusted more.
        vr.random_ok = vr.hard_ok and candidate.path.value in {"P1_tool", "hard_mode"}

        if normalized is not None:
            vr.judge_prob = self.judge.score(problem, candidate.trace, normalized, artifacts)
        else:
            vr.judge_prob = 0.0

        contradiction = bool(not vr.hard_ok)
        vr.contradiction = contradiction

        score = 0.0
        score += self.weights.hard_constraints * float(vr.hard_ok)
        score += self.weights.symbolic_consistency * float(vr.symbolic_ok)
        score += self.weights.randomized_tests * float(vr.random_ok)
        score += self.weights.judge_prob * float(vr.judge_prob)
        score += self.weights.self_consistency * vr.self_consistency_vote_share
        if contradiction:
            score += self.weights.contradiction_penalty
        if vr.sandbox_error:
            score += self.weights.sandbox_penalty
        vr.score = score
        vr.artifacts = artifacts
        return vr

    def verify_batch(self, candidates: list[Candidate], problem: str, meta: ProblemMetadata) -> list[VerificationResult]:
        return [self.verify_candidate(c, problem, meta) for c in candidates]

    def apply_vote_share(self, verified: list[VerificationResult]) -> None:
        valid_answers = [v.normalized_answer for v in verified if v.hard_ok and v.normalized_answer is not None]
        if not valid_answers:
            return
        counts = Counter(valid_answers)
        total = sum(counts.values())
        for item in verified:
            if item.normalized_answer is None:
                item.self_consistency_vote_share = 0.0
            else:
                item.self_consistency_vote_share = counts[item.normalized_answer] / total
            item.score += self.weights.self_consistency * item.self_consistency_vote_share

    def confident_enough(self, verified: list[VerificationResult]) -> bool:
        if not verified:
            return False
        self.apply_vote_share(verified)
        best = max(verified, key=lambda x: x.score)
        return (
            best.score >= self.config.confidence_threshold
            and best.self_consistency_vote_share >= self.config.vote_share_threshold
        )

    def select_final(self, verified: list[VerificationResult]) -> tuple[VerificationResult | None, str]:
        if not verified:
            return None, "no_candidates"
        self.apply_vote_share(verified)
        sorted_verified = sorted(
            verified,
            key=lambda x: (x.score, x.self_consistency_vote_share, x.judge_prob),
            reverse=True,
        )
        best = sorted_verified[0]
        reason = "top_score_after_verification"
        return best, reason
