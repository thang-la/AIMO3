from __future__ import annotations

from collections import Counter, defaultdict

from aimo3.config import SolverConfig, VerificationWeights
from aimo3.llm import NeuralJudge
from aimo3.models import Candidate, ProblemMetadata, VerificationResult
from aimo3.sandbox import run_python_sandbox


class Verifier:
    def __init__(self, config: SolverConfig, judge: NeuralJudge):
        self.config = config
        self.weights = VerificationWeights()
        self.judge = judge

    def _normalize_answer(self, ans: int | None, meta: ProblemMetadata, candidate: Candidate) -> int | None:
        if ans is None:
            return None
        skip_mod = bool(candidate.metadata.get("skip_modulus_normalization"))
        if not skip_mod and meta.modulus is not None and meta.modulus > 0:
            ans = ans % meta.modulus
        ans = ans % self.config.global_modulus_fallback
        return int(ans)

    def _check_range(self, ans: int | None) -> bool:
        if ans is None:
            return False
        return 0 <= ans <= self.config.answer_upper_bound

    def _run_validator(self, validator_code: str, normalized_answer: int) -> tuple[bool | None, dict]:
        wrapper = (
            f"CANDIDATE_ANSWER = {normalized_answer}\n"
            f"{validator_code}\n"
            "if 'IS_VALID' in locals():\n"
            "    ANSWER = int(bool(IS_VALID))\n"
            "elif 'VALID' in locals():\n"
            "    ANSWER = int(bool(VALID))\n"
            "elif 'ANSWER_OK' in locals():\n"
            "    ANSWER = int(bool(ANSWER_OK))\n"
            "else:\n"
            "    ANSWER = 1\n"
        )
        result = run_python_sandbox(
            wrapper,
            timeout_s=self.config.sandbox_timeout_s,
            memory_mb=self.config.sandbox_memory_mb,
        )
        artifacts = {
            "success": result.success,
            "timeout": result.timeout,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "error": result.error,
            "answer": result.answer,
        }
        if not result.success or result.answer is None:
            return None, artifacts
        return bool(result.answer), artifacts

    def _base_score(self, vr: VerificationResult) -> float:
        score = 0.0
        score += self.weights.hard_constraints * float(vr.hard_ok)
        score += self.weights.symbolic_consistency * float(vr.symbolic_ok)
        score += self.weights.randomized_tests * float(vr.random_ok)
        score += self.weights.judge_prob * float(vr.judge_prob)
        if vr.contradiction:
            score += self.weights.contradiction_penalty
        if vr.sandbox_error:
            score += self.weights.sandbox_penalty
        if vr.artifacts.get("validator_ok"):
            score += self.weights.validator_bonus
        conf_hint = vr.candidate.metadata.get("confidence_hint")
        if isinstance(conf_hint, (int, float)):
            score += 0.25 * (float(conf_hint) - 0.5)
        return score

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
            artifacts["sandbox_timeout"] = sandbox.timeout
            if sandbox.success:
                answer = sandbox.answer
                artifacts["tool_ok"] = True
            else:
                artifacts["tool_ok"] = False
                vr.sandbox_error = True
                vr.notes.append("sandbox execution failed")

        normalized = self._normalize_answer(answer, meta, candidate)
        vr.normalized_answer = normalized
        vr.hard_ok = self._check_range(normalized)
        if not vr.hard_ok:
            vr.notes.append("hard constraints failed (range/format)")

        validator_ok: bool | None = None
        validator_code = candidate.metadata.get("validator_code")
        if isinstance(validator_code, str) and validator_code.strip() and normalized is not None:
            validator_ok, validator_artifacts = self._run_validator(validator_code, normalized)
            artifacts["validator"] = validator_artifacts
            if validator_ok is True:
                artifacts["validator_ok"] = True
                vr.notes.append("validator accepted")
            elif validator_ok is False:
                artifacts["validator_ok"] = False
                vr.contradiction = True
                vr.notes.append("validator rejected")
            else:
                vr.notes.append("validator failed")

        vr.symbolic_ok = vr.hard_ok and (len(meta.extracted_equations) == 0 or normalized is not None)
        if validator_ok is True:
            vr.symbolic_ok = True

        vr.random_ok = vr.hard_ok and bool(artifacts.get("tool_ok") or artifacts.get("validator_ok"))
        if not vr.random_ok and candidate.path.value in {"P2_reasoning", "P3_backsolve"} and vr.hard_ok:
            vr.random_ok = True

        if normalized is not None:
            vr.judge_prob = self.judge.score(problem, candidate.trace, normalized, artifacts)
        else:
            vr.judge_prob = 0.0

        if not vr.hard_ok:
            vr.contradiction = True

        vr.artifacts = artifacts
        base = self._base_score(vr)
        vr.artifacts["_base_score"] = base
        vr.score = base
        return vr

    def verify_batch(self, candidates: list[Candidate], problem: str, meta: ProblemMetadata) -> list[VerificationResult]:
        return [self.verify_candidate(c, problem, meta) for c in candidates]

    def _apply_vote_share(self, verified: list[VerificationResult]) -> None:
        valid_answers = [v.normalized_answer for v in verified if v.hard_ok and v.normalized_answer is not None]
        if not valid_answers:
            for item in verified:
                item.self_consistency_vote_share = 0.0
            return
        counts = Counter(valid_answers)
        total = sum(counts.values())
        for item in verified:
            if item.normalized_answer is None:
                item.self_consistency_vote_share = 0.0
            else:
                item.self_consistency_vote_share = counts[item.normalized_answer] / total

    def _apply_path_diversity(self, verified: list[VerificationResult]) -> None:
        valid = [v for v in verified if v.hard_ok and v.normalized_answer is not None]
        if not valid:
            for item in verified:
                item.path_diversity = 0.0
            return
        answer_paths: dict[int, set[str]] = defaultdict(set)
        all_paths: set[str] = set()
        for item in valid:
            assert item.normalized_answer is not None
            answer_paths[item.normalized_answer].add(item.candidate.path.value)
            all_paths.add(item.candidate.path.value)
        total_paths = max(1, len(all_paths))
        for item in verified:
            if item.normalized_answer is None:
                item.path_diversity = 0.0
            else:
                item.path_diversity = len(answer_paths[item.normalized_answer]) / total_paths

    def _recompute_total_score(self, verified: list[VerificationResult]) -> None:
        self._apply_vote_share(verified)
        self._apply_path_diversity(verified)
        for item in verified:
            base = float(item.artifacts.get("_base_score", item.score))
            item.score = (
                base
                + self.weights.self_consistency * item.self_consistency_vote_share
                + self.weights.path_diversity * item.path_diversity
            )

    def confident_enough(self, verified: list[VerificationResult]) -> bool:
        if not verified:
            return False
        self._recompute_total_score(verified)
        best = max(verified, key=lambda x: x.score)
        diverse_ok = True
        if self.config.require_path_diversity_for_confidence:
            diverse_ok = best.path_diversity >= 0.34
        return (
            best.score >= self.config.confidence_threshold
            and best.self_consistency_vote_share >= self.config.vote_share_threshold
            and diverse_ok
        )

    def select_final(self, verified: list[VerificationResult]) -> tuple[VerificationResult | None, str]:
        if not verified:
            return None, "no_candidates"
        self._recompute_total_score(verified)
        sorted_verified = sorted(
            verified,
            key=lambda x: (
                x.score,
                x.path_diversity,
                x.self_consistency_vote_share,
                bool(x.artifacts.get("validator_ok")),
                x.judge_prob,
            ),
            reverse=True,
        )
        return sorted_verified[0], "top_score_after_verification"
