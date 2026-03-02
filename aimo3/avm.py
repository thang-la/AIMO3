from __future__ import annotations

from .types import BeliefState
from .utils import clamp, top2


class ActionValueModel:
    """Operational EVI predictor used by controller."""

    def predict(self, st: BeliefState, act: dict) -> dict[str, float]:
        (_, p1), (_, p2) = top2(st.pi)
        margin = p1 - p2
        frag_top = st.diagnostics.get("fragile_top", 0.5)
        diversity = st.diagnostics.get("cluster_diversity", 1)
        hardness = st.diagnostics.get("hardness", st.meta.get("difficulty", 0.5))

        cost = 1.0
        timeout = 0.02
        delta = 0.0

        t = act["type"]
        if t == "GEN":
            cost = 1.0
            timeout = 0.03 + 0.05 * hardness
            delta = 0.05 * (1.0 - margin) * (1.2 if diversity < 2 else 0.8)
        elif t == "VERIFY":
            cost = 0.6
            timeout = 0.02
            delta = 0.04 * frag_top + 0.03 * (1.0 - margin)
        elif t == "REFUTE":
            cost = 0.8
            timeout = 0.03
            delta = 0.08 * frag_top
        elif t == "DISAMBIGUATE":
            cost = 0.9
            timeout = 0.03
            delta = 0.09 * max(0.0, 0.15 - margin) / 0.15
        elif t == "REPARSE":
            cost = 0.5
            timeout = 0.01
            delta = 0.06 * st.diagnostics.get("misparse_risk", 0.0)
        elif t in {"HYP_START", "HYP_STEP"}:
            cost = 1.2
            timeout = 0.06
            delta = 0.07 * hardness * (1.0 - margin)

        if st.budget.time_left() < 15:
            delta *= 0.6
            timeout *= 1.4

        return {
            "E_deltaV": clamp(delta, -1.0, 1.0),
            "cost": clamp(cost, 0.1, 10.0),
            "timeout_risk": clamp(timeout, 0.0, 1.0),
        }

