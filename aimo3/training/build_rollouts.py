from __future__ import annotations

import json
from pathlib import Path

from ..controller import AIMO3Solver
from ..training.records import CandidateRecord, DecisionStepRecord, PairRecord, RolloutRecord
from ..utils import hash64
from ..ced import pair_features


def build_rollout(problem_id: str, latex: str, truth_answer: int, run_seed: int | None = None) -> RolloutRecord:
    seed = run_seed if run_seed is not None else int(hash64(problem_id) & 0xFFFFFFFF)
    solver = AIMO3Solver()
    ans = solver.solve_one(problem_id, latex, seed)
    del ans

    # A fresh state to extract internal traces with deterministic actions.
    st = solver.init_state(latex, seed)
    for act in [{"type": "GEN", "policy": p, "n": 1} for p in st.run_config["allowed_policies"][:2]]:
        solver.execute_action_and_update(st, act, seed)

    rollout = RolloutRecord(problem_id=problem_id, run_program=st.run_config["program"])

    for c in st.candidates:
        feats = {
            "arm_p": c.arm_p,
            "arm_u": c.arm_u,
            "redundancy_w": c.redundancy_w,
            "cluster_id": float(c.cluster_id),
            "difficulty": float(st.meta.get("difficulty", 0.5)),
        }
        ft = max(c.failure_logits.items(), key=lambda kv: kv[1])[0] if c.failure_logits else ""
        rollout.candidate_records.append(
            CandidateRecord(
                problem_id=problem_id,
                candidate_id=c.id,
                features=feats,
                label=int(c.answer == truth_answer),
                failure_type=ft,
                policy_id=c.policy_id,
                answer=c.answer,
            )
        )

    for i in range(len(st.candidates)):
        for j in range(i + 1, len(st.candidates)):
            ci, cj = st.candidates[i], st.candidates[j]
            rollout.pair_records.append(
                PairRecord(
                    problem_id=problem_id,
                    i=ci.id,
                    j=cj.id,
                    pair_features=pair_features(ci, cj),
                    p_i=ci.arm_p,
                    p_j=cj.arm_p,
                    y_i=int(ci.answer == truth_answer),
                    y_j=int(cj.answer == truth_answer),
                )
            )

    prev = 0.0
    for t, h in enumerate(st.action_history):
        pi_top = float(h.get("pi_top", 0.0))
        rollout.decision_steps.append(
            DecisionStepRecord(
                problem_id=problem_id,
                t=t,
                state_features={
                    "pi_top": pi_top,
                    "time_left": float(h.get("time_left", 0.0)),
                    "evi": float(h.get("evi", 0.0)),
                },
                action=h.get("act", {}),
                outcome={"pi_top": pi_top},
                delta_v=pi_top - prev,
                cost=max(0.0, 1.0 - float(h.get("time_left", 0.0))),
                timeout=0,
            )
        )
        prev = pi_top

    return rollout


def build_rollouts_from_jsonl(dataset_path: str | Path, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for line in Path(dataset_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rr = build_rollout(str(row["id"]), row["problem"], int(row["answer"]))
        p = out / f"{row['id']}.json"
        p.write_text(json.dumps(rr, default=lambda o: o.__dict__, ensure_ascii=True, indent=2), encoding="utf-8")


if __name__ == "__main__":  # pragma: no cover
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("out_dir")
    args = ap.parse_args()
    build_rollouts_from_jsonl(args.dataset, args.out_dir)

