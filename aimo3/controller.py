from __future__ import annotations

import os
import random
import time
from dataclasses import asdict
from typing import Any

from .aas import select_answer
from .actions import enumerate_actions, seed_action_set
from .arm import ARMModel, build_arm_features
from .aspr import adversarial_second_pass, run_refutation
from .avm import ActionValueModel
from .baa import baa_posterior
from .belief import (
    apply_evidence,
    compute_baa_posterior,
    init_belief_state_scores,
    integrate_candidate_support,
    integrate_disambiguation_online,
    integrate_hard_constraint_refutation,
    integrate_refutation_online,
    top_answers,
    update_posterior,
    update_uncertainty_by_answer,
)
from .budget import allocate_budget
from .ced import assign_clusters_and_weights, answer_cluster_support
from .config import DEFAULT_CONFIG, SystemConfig
from .diagnostics import init_diagnostics, update_diagnostics
from .disambiguate import run_disambiguation_tests
from .esmp import esmp_next_action, evi, should_stop
from .fallback import fallback_logic, should_fallback
from .hypothesis import apply_hypothesis_evidence, hypothesis_step, init_hypotheses
from .models import (
    BaseGeneratorModel,
    BaseJudgeModel,
    DeterministicJudge,
    load_required_gpt_oss_120b,
)
from .parser import parse
from .policies import CandidateFactory, run_policy_generate
from .rde import adapt_run_config_for_low_diversity, select_run_config
from .router import route
from .types import BeliefState, Candidate, Evidence
from .utils import hash64
from .verifier import verify_deep, verify_shallow


class AIMO3Solver:
    def __init__(
        self,
        *,
        cfg: SystemConfig | None = None,
        main_model: BaseGeneratorModel | None = None,
        fast_model: BaseGeneratorModel | None = None,
        judge_model: BaseJudgeModel | None = None,
        arm_model: ARMModel | None = None,
        avm_model: ActionValueModel | None = None,
    ) -> None:
        self.cfg = cfg or DEFAULT_CONFIG
        self.main_model = main_model or load_required_gpt_oss_120b()
        self.fast_model = fast_model or self.main_model
        self.judge_model = judge_model or DeterministicJudge()
        self.arm_model = arm_model or ARMModel()
        self.avm_model = avm_model or ActionValueModel()
        self.candidate_factory = CandidateFactory()

    def solve_one(self, pid: str, latex: str, run_seed: int | None = None) -> int:
        seed = run_seed if run_seed is not None else int.from_bytes(os.urandom(8), "big")
        st = self.init_state(latex, seed)

        for act in seed_action_set(st):
            self.execute_action_and_update(st, act, seed)
            if should_stop(st):
                return self.output_answer(pid, st, seed)

        while st.budget.time_left() > 0:
            update_posterior(st)
            self.apply_baa_projection(st)
            st.diagnostics = update_diagnostics(st)
            st.run_config = adapt_run_config_for_low_diversity(st.run_config, st.diagnostics)

            if should_stop(st):
                return self.output_answer(pid, st, seed)

            actions = enumerate_actions(st)
            best_act, best_evi = esmp_next_action(st, actions, self.avm_model)
            if not best_act:
                return self.output_answer(pid, st, seed)
            if best_evi <= st.run_config.get("evi_stop_floor", 0.0):
                return self.output_answer(pid, st, seed)

            self.execute_action_and_update(st, best_act, seed)

        return self.output_answer(pid, st, seed)

    def init_state(self, latex: str, run_seed: int) -> BeliefState:
        meta, constraints = parse(latex, strictness="normal")
        base_mode = route(meta, latex)
        run_config = select_run_config(run_seed, meta.get("problem_hash", "0"), meta, base_mode, self.cfg)
        budget = allocate_budget(meta, base_mode, run_config, self.cfg)

        s, pi, u = init_belief_state_scores(self.cfg)
        diagnostics = init_diagnostics(meta, constraints, run_config)
        diagnostics["rng"] = random.Random(hash64(f"{run_seed}:{meta.get('problem_hash', '0')}") & 0xFFFFFFFF)

        return BeliefState(
            s=s,
            pi=pi,
            u=u,
            clusters={},
            candidates=[],
            meta=meta,
            constraints=constraints,
            run_config=run_config,
            budget=budget,
            diagnostics=diagnostics,
            action_history=[],
        )

    def execute_action_and_update(self, st: BeliefState, act: dict, run_seed: int) -> None:
        t0 = time.monotonic()
        p_prev = max(st.pi.values()) if st.pi else 0.0

        if act["type"] == "GEN":
            model = self.main_model if st.run_config.get("program") == "A" else self.fast_model
            new_cands = run_policy_generate(
                model,
                self.candidate_factory,
                act["policy"],
                st.meta["raw_latex"],
                st.meta,
                n=act.get("n", 1),
                base_seed=run_seed,
            )
            for cand in new_cands:
                self.integrate_candidate_online(st, cand)

        elif act["type"] == "VERIFY":
            cand = self._get_candidate(st, act["cand_id"])
            if cand is not None:
                self.integrate_candidate_evidence_update(st, cand, level=act.get("level", "deep"))

        elif act["type"] == "REFUTE":
            ev = run_refutation(st, act["answer"], mode=act.get("mode", "mild"))
            integrate_refutation_online(st, ev, self.cfg)

        elif act["type"] == "DISAMBIGUATE":
            ev = run_disambiguation_tests(st, act["a"], act["b"])
            integrate_disambiguation_online(st, ev)

        elif act["type"] == "REPARSE":
            meta2, constraints2 = parse(st.meta["raw_latex"], strictness=act.get("strictness", "strict"))
            st.meta = {**st.meta, **meta2}
            st.constraints = constraints2
            st.diagnostics["misparse_risk"] = 0.2

        elif act["type"] == "HYP_START":
            st.diagnostics["hyp_started"] = True
            st.diagnostics["hypotheses"] = init_hypotheses(st.meta, k=act.get("k", st.run_config.get("hyp_k", 3)))

        elif act["type"] == "HYP_STEP":
            hyps = st.diagnostics.get("hypotheses", {})
            hyp = hyps.get(act["hyp_id"])
            if hyp is not None:
                evs = hypothesis_step(st, hyp, op=act["op"])
                apply_hypothesis_evidence(st, hyp, evs)
                for ev in evs:
                    apply_evidence(st, ev)
                if act["op"] == "MODEL":
                    policy = self._policy_from_hyp_tag(getattr(hyp, "tag", ""))
                    new_cands = run_policy_generate(
                        self.main_model,
                        self.candidate_factory,
                        policy,
                        st.meta["raw_latex"],
                        st.meta,
                        n=1,
                        base_seed=run_seed + act["hyp_id"],
                    )
                    for cand in new_cands:
                        self.integrate_candidate_online(st, cand)

        # optional ASPR injection for fragility or run B
        st.diagnostics = update_diagnostics(st)
        if st.run_config.get("adversarial") or st.diagnostics.get("fragile_top", 1.0) > 0.6:
            asp_cands = adversarial_second_pass(
                self.fast_model,
                self.candidate_factory,
                st.meta["raw_latex"],
                st.meta,
                st.run_config,
                st.diagnostics,
                base_seed=run_seed,
            )
            for cand in asp_cands[:2]:
                self.integrate_candidate_online(st, cand)

        update_posterior(st)
        self.apply_baa_projection(st)
        st.diagnostics = update_diagnostics(st)

        dt = max(1e-3, time.monotonic() - t0)
        st.action_history.append(
            {
                "act": act,
                "pi_top_prev": p_prev,
                "pi_top": max(st.pi.values()) if st.pi else 0.0,
                "time_left": st.budget.time_left(),
                "delta_t": dt,
                "evi": evi(st, act, self.avm_model),
            }
        )
        st.diagnostics["action_counts"][act["type"]] += 1

    def integrate_candidate_online(self, st: BeliefState, cand: Candidate) -> None:
        shallow = verify_shallow(cand, st.meta, st.constraints)
        if not shallow.hard_ok:
            cand.verifier = {"shallow": asdict(shallow), "deep_done": False}
            cand.arm_logbf = -self.cfg.belief.hard_refute_M
            cand.arm_p = 1e-6
            cand.arm_u = 0.0
            st.candidates.append(cand)
            self.recompute_clusters_arm_and_belief(st)
            integrate_hard_constraint_refutation(st, cand, self.cfg)
            return

        cand.verifier = {"shallow": asdict(shallow), "deep_done": False}
        st.candidates.append(cand)
        self.recompute_clusters_arm_and_belief(st)

    def integrate_candidate_evidence_update(self, st: BeliefState, cand: Candidate, level: str = "deep") -> None:
        if level == "deep":
            deep = verify_deep(cand, st.meta, st.constraints, timeout_s=self.cfg.budget.sandbox_seconds_per_run)
            cand.verifier["deep"] = asdict(deep)
            cand.verifier["deep_done"] = True
            # Recompute support evidence from ARM features with deep verification signals.
            self.recompute_clusters_arm_and_belief(st)
            update_posterior(st)
            self.apply_baa_projection(st)

    def recompute_clusters_arm_and_belief(self, st: BeliefState) -> None:
        rho, clusters, _weights = assign_clusters_and_weights(st.candidates)
        st.clusters = self._build_cluster_meta(st, clusters)
        support = answer_cluster_support(st.candidates, clusters)

        # Keep non-support evidence, then rebuild support evidence from current candidates.
        prev_support_scores: dict[int | str, float] = st.diagnostics.get("support_scores", {})
        base_scores: dict[int | str, float] = {}
        for key in set(st.s.keys()) | set(prev_support_scores.keys()):
            val = st.s.get(key, 0.0) - prev_support_scores.get(key, 0.0)
            if key == "OTHER" or abs(val) > 1e-12:
                base_scores[key] = val
        if "OTHER" not in base_scores:
            base_scores["OTHER"] = self.cfg.belief.other_log_prior
        st.s = base_scores

        arm_ps: list[float] = []
        support_scores: dict[int | str, float] = {}
        for c in st.candidates:
            shallow = c.verifier.get("shallow", {})
            deep = c.verifier.get("deep", {})
            deep_like = {
                "hard_ok": bool(shallow.get("hard_ok", False)),
                "symbolic_ok": bool(deep.get("symbolic_ok", False)),
                "random_ok_rate": float(deep.get("random_ok_rate", 0.0)),
                "tool_ok": bool(deep.get("tool_ok", False)),
                "artifacts": deep.get("artifacts", {}),
                "contradictions": int(deep.get("contradictions", 0)),
            }
            deep_like["judge_prob"] = float(
                self.judge_model.score(
                    st.meta.get("raw_latex", ""),
                    c.trace,
                    c.answer,
                    deep_like["artifacts"],
                )
            )
            fake_verify = self._verification_proxy(c.answer, deep_like)
            feats = build_arm_features(st.meta, c, fake_verify, st.candidates, support)
            arm = self.arm_model.predict(feats)
            c.arm_logbf = arm.logbf_support
            c.arm_p = arm.p_correct
            c.arm_u = arm.uncertainty
            c.failure_logits = arm.failure_logits
            arm_ps.append(arm.p_correct)
            before = st.s.get(c.answer, 0.0)
            integrate_candidate_support(st, c, self.cfg)
            support_scores[c.answer] = support_scores.get(c.answer, 0.0) + (st.s.get(c.answer, 0.0) - before)

        update_uncertainty_by_answer(st)
        update_posterior(st)

        st.diagnostics["rho"] = rho
        st.diagnostics["clusters_raw"] = clusters
        st.diagnostics["arm_ps"] = arm_ps
        st.diagnostics["support_scores"] = support_scores

    def apply_baa_projection(self, st: BeliefState) -> None:
        if not st.candidates:
            update_posterior(st)
            return
        rho = st.diagnostics.get("rho", [])
        clusters = st.diagnostics.get("clusters_raw", {})
        arm_ps = st.diagnostics.get("arm_ps", [c.arm_p for c in st.candidates])
        pi_baa = baa_posterior(st.candidates, arm_ps, rho, clusters)

        # Combine dynamic evidence (st.pi) and BAA posterior via multiplicative pooling.
        pi_dyn = compute_baa_posterior(st)
        all_keys = set(pi_dyn.keys()) | set(pi_baa.keys())
        combined: dict[int | str, float] = {}
        for k in all_keys:
            combined[k] = max(1e-12, pi_dyn.get(k, 1e-12)) * max(1e-12, pi_baa.get(k, 1e-12))
        z = sum(combined.values())
        if z > 0:
            st.pi = {k: v / z for k, v in combined.items()}
        st.diagnostics["pi_baa"] = pi_baa

    def output_answer(self, pid: str, st: BeliefState, run_seed: int) -> int:
        update_posterior(st)
        self.apply_baa_projection(st)
        st.diagnostics = update_diagnostics(st)

        rng = st.diagnostics["rng"]
        answer, decision_meta = select_answer(st, self.cfg, rng)
        st.diagnostics["decision_meta"] = decision_meta

        if should_fallback(answer, st.diagnostics):
            answer = fallback_logic(st.meta, st.diagnostics, st.candidates, rng)

        answer = self._apply_modulus_and_range(answer, st.meta)
        st.diagnostics["final_answer"] = answer
        st.diagnostics["pid"] = pid
        return int(answer)

    def _get_candidate(self, st: BeliefState, cid: int) -> Candidate | None:
        for c in st.candidates:
            if c.id == cid:
                return c
        return None

    def _build_cluster_meta(self, st: BeliefState, clusters: dict[int, list[int]]) -> dict[int, Any]:
        out: dict[int, Any] = {}
        for cid, members in clusters.items():
            lemmas = []
            modes = []
            tool_sig = []
            for i in members:
                c = st.candidates[i]
                modes.append(c.policy_id)
                lemmas.extend(c.trace_summary.lower().split()[:4])
                if c.tool_code:
                    tool_sig.append(c.tool_code[:80])
            out[cid] = {
                "cluster_id": cid,
                "member_ids": [st.candidates[i].id for i in members],
                "cluster_mode": max(set(modes), key=modes.count) if modes else "unknown",
                "cluster_weight": sum(st.candidates[i].redundancy_w for i in members),
                "top_lemmas": lemmas[:8],
                "tool_signature": "|".join(tool_sig[:2]),
            }
        return out

    def _verification_proxy(self, answer: int, d: dict[str, Any]):
        class _V:
            pass

        v = _V()
        v.answer = answer
        v.hard_ok = bool(d.get("hard_ok", False))
        v.symbolic_ok = bool(d.get("symbolic_ok", False))
        v.random_ok_rate = float(d.get("random_ok_rate", 0.0))
        v.tool_ok = bool(d.get("tool_ok", False))
        v.artifacts = d.get("artifacts", {})
        v.contradictions = int(d.get("contradictions", 0))
        v.judge_prob = float(d.get("judge_prob", 0.0))
        return v

    def _policy_from_hyp_tag(self, tag: str) -> str:
        if tag.startswith("GEO"):
            return "GEO_COORD"
        if tag.startswith("COMB"):
            return "COMB_SMALLBRUTE"
        if tag.startswith("NT") or tag.startswith("ALG"):
            return "A_SYMBOLIC"
        return "B_INVARIANT"

    def _apply_modulus_and_range(self, ans: int, meta: dict) -> int:
        mod = meta.get("modulus")
        val = int(ans)
        if mod is not None and int(mod) > 0:
            val = val % int(mod)
        return val % 100000
