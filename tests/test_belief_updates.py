from __future__ import annotations

from aimo3.belief import apply_evidence, init_belief_state_scores, update_posterior
from aimo3.config import DEFAULT_CONFIG
from aimo3.types import BeliefState, Budget, Cluster, Evidence


def make_state() -> BeliefState:
    s, pi, u = init_belief_state_scores(DEFAULT_CONFIG)
    return BeliefState(
        s=s,
        pi=pi,
        u=u,
        clusters={},
        candidates=[],
        meta={},
        constraints={},
        run_config={"stop_pi": 0.9, "min_clusters": 1, "max_fragility": 0.5},
        budget=Budget(total_seconds=10, token_budget=1000, sandbox_runs_left=1),
        diagnostics={},
        action_history=[],
    )


def test_support_and_refute_update_scores() -> None:
    st = make_state()
    apply_evidence(st, Evidence(type="support", target=12345, logbf=2.0, weight=1.0))
    apply_evidence(st, Evidence(type="support", target=54321, logbf=0.5, weight=1.0))
    update_posterior(st)
    assert st.pi[12345] > st.pi[54321]

    apply_evidence(st, Evidence(type="refute", target=12345, logbf=-10.0, weight=1.0))
    update_posterior(st)
    assert st.pi[12345] < st.pi[54321]


def test_disambiguation_updates_both_answers() -> None:
    st = make_state()
    st.s[10] = -1.0
    st.s[20] = -1.0
    apply_evidence(st, Evidence(type="disambiguate", target=(10, 20), logbf=1.5, weight=1.0))
    update_posterior(st)
    assert st.pi[10] > st.pi[20]

