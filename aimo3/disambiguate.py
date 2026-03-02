from __future__ import annotations

from .types import DisambiguationEvent


def run_disambiguation_tests(st, a: int, b: int) -> DisambiguationEvent:
    # Approximate separating test signal from support/fragility asymmetry.
    sa = st.s.get(a, -20.0)
    sb = st.s.get(b, -20.0)
    fa = st.diagnostics.get("fragility", {}).get(a, 0.7)
    fb = st.diagnostics.get("fragility", {}).get(b, 0.7)
    raw = (sa - sb) + 0.8 * (fb - fa)
    logbf = max(-3.0, min(3.0, raw))
    reliability = max(0.1, min(1.0, 1.0 - 0.5 * (fa + fb)))
    return DisambiguationEvent(a=a, b=b, logbf_ab=logbf, reliability=reliability, payload={"raw": raw})

