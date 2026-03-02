from __future__ import annotations

from aimo3.ced import assign_clusters_and_weights
from aimo3.types import Candidate


def _cand(i: int, answer: int, policy: str, trace: str, code: str) -> Candidate:
    return Candidate(
        id=i,
        answer=answer,
        policy_id=policy,
        path_type="P1",
        trace=trace,
        trace_summary=trace,
        tool_code=code,
    )


def test_ced_clusters_redundant_candidates() -> None:
    cands = [
        _cand(0, 111, "A_TOOLFORMAL", "use crt and mod", "ANSWER=111"),
        _cand(1, 111, "A_TOOLFORMAL", "use crt and mod", "ANSWER=111"),
        _cand(2, 222, "B_INVARIANT", "invariant parity contradiction", "ANSWER=222"),
    ]
    rho, clusters, weights = assign_clusters_and_weights(cands)
    assert len(clusters) >= 2
    assert cands[0].cluster_id == cands[1].cluster_id
    assert weights[0] < 1.0
    assert rho[0][1] > rho[0][2]

