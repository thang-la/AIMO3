from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .records import CandidateRecord, DecisionStepRecord, PairRecord, RolloutRecord


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def export_rollout(path: str | Path, rollout: RolloutRecord) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "problem_id": rollout.problem_id,
        "run_program": rollout.run_program,
        "candidate_records": [asdict(r) for r in rollout.candidate_records],
        "pair_records": [asdict(r) for r in rollout.pair_records],
        "decision_steps": [asdict(r) for r in rollout.decision_steps],
    }
    p.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def flatten_candidate_records(rollouts: list[RolloutRecord]) -> list[dict]:
    rows: list[dict] = []
    for r in rollouts:
        for c in r.candidate_records:
            row = asdict(c)
            row["problem_id"] = r.problem_id
            rows.append(row)
    return rows


def flatten_pair_records(rollouts: list[RolloutRecord]) -> list[dict]:
    rows: list[dict] = []
    for r in rollouts:
        for p in r.pair_records:
            row = asdict(p)
            row["problem_id"] = r.problem_id
            rows.append(row)
    return rows


def flatten_decision_steps(rollouts: list[RolloutRecord]) -> list[dict]:
    rows: list[dict] = []
    for r in rollouts:
        for s in r.decision_steps:
            row = asdict(s)
            row["problem_id"] = r.problem_id
            rows.append(row)
    return rows

