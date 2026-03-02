from __future__ import annotations

from dataclasses import replace

from aimo3.config import SolverConfig
from aimo3.controller import AIMO3Solver
from aimo3.runtime import solver_config_from_env


def run_self_play(records: list[dict], solver: AIMO3Solver | None = None) -> list[dict]:
    if solver is None:
        config: SolverConfig = solver_config_from_env()
        if config.llm.backend == "heuristic":
            config = replace(config, allow_demo_fallback=True, enforce_real_backend=False)
        solver = AIMO3Solver(config=config)
    out: list[dict] = []
    for idx, row in enumerate(records):
        prompt = str(row.get("prompt", ""))
        answer = row.get("response", {}).get("final", {}).get("answer")  # type: ignore[union-attr]
        result = solver.solve_one(pid=f"selfplay-{idx}", latex=prompt)
        predicted = int(result.answer)
        correct = int(answer) == predicted if answer is not None else False
        out.append(
            {
                "id": f"selfplay-{idx}",
                "prompt": prompt,
                "gold_answer": answer,
                "pred_answer": predicted,
                "correct": bool(correct),
                "route": {
                    "domain": result.route.domain.value,
                    "difficulty": result.route.difficulty.value,
                },
                "candidates": [
                    {
                        "path": c.candidate.path.value,
                        "answer": c.normalized_answer,
                        "score": c.score,
                        "hard_ok": c.hard_ok,
                    }
                    for c in result.candidates
                ],
            }
        )
    return out
