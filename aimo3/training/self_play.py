from __future__ import annotations

from aimo3.controller import AIMO3Solver


def run_self_play(records: list[dict], solver: AIMO3Solver | None = None) -> list[dict]:
    solver = solver or AIMO3Solver()
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
