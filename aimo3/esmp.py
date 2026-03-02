from __future__ import annotations

from .avm import ActionValueModel
from .belief import expected_correctness_if_stop
from .types import BeliefState
from .utils import top2


def evi(st: BeliefState, act: dict, avm: ActionValueModel) -> float:
    pred = avm.predict(st, act)
    return (
        pred["E_deltaV"]
        - st.run_config["lambda_cost"] * pred["cost"]
        - st.run_config["eta_timeout"] * pred["timeout_risk"]
    )


def esmp_next_action(st: BeliefState, actions: list[dict], avm: ActionValueModel) -> tuple[dict | None, float]:
    if not actions:
        return None, 0.0
    best = max(actions, key=lambda a: evi(st, a, avm))
    best_evi = evi(st, best, avm)
    return best, best_evi


def should_stop(st: BeliefState) -> bool:
    a1, p1 = max(st.pi.items(), key=lambda kv: kv[1])
    cluster_support = st.diagnostics.get("cluster_support", {}).get(a1, 0)
    fragility = st.diagnostics.get("fragility", {}).get(a1, 1.0)
    if (
        p1 >= st.run_config["stop_pi"]
        and cluster_support >= st.run_config["min_clusters"]
        and fragility <= st.run_config["max_fragility"]
    ):
        return True

    # hard-mode diminishing returns stop
    if st.diagnostics.get("gain_rate_streak", 0) >= 3 and st.diagnostics.get("gain_rate", 1.0) < 0.002:
        return True

    if st.budget.time_left() <= 0.0:
        return True

    return False


def need_refute(st: BeliefState, a1: int | str) -> bool:
    if a1 == "OTHER":
        return False
    return (
        st.diagnostics.get("fragility", {}).get(a1, 1.0) > 0.35
        or st.diagnostics.get("cluster_support", {}).get(a1, 0) <= 1
    )


def is_split(st: BeliefState, a1: int | str, a2: int | str) -> bool:
    if a1 == "OTHER" or a2 == "OTHER":
        return False
    p1 = st.pi.get(a1, 0.0)
    p2 = st.pi.get(a2, 0.0)
    return p1 < 0.75 and (p1 - p2) < 0.10


def should_refute_by_esmp(st: BeliefState) -> bool:
    (a1, p1), (a2, p2) = top2(st.pi)
    if a1 == "OTHER":
        return False
    if p1 - p2 < 0.12:
        return True
    return need_refute(st, a1)


def value(st: BeliefState) -> float:
    return expected_correctness_if_stop(st)

