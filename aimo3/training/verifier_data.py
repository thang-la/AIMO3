from __future__ import annotations


def build_verifier_pairs(self_play_rollouts: list[dict]) -> list[dict]:
    pairs: list[dict] = []
    for row in self_play_rollouts:
        prompt = row["prompt"]
        candidates = row.get("candidates", [])
        good = [c for c in candidates if c.get("hard_ok")]
        bad = [c for c in candidates if not c.get("hard_ok")]
        for g in good[:3]:
            for b in bad[:3]:
                pairs.append(
                    {
                        "prompt": prompt,
                        "chosen": {
                            "answer": g.get("answer"),
                            "score": g.get("score"),
                            "path": g.get("path"),
                        },
                        "rejected": {
                            "answer": b.get("answer"),
                            "score": b.get("score"),
                            "path": b.get("path"),
                        },
                    }
                )
    return pairs
