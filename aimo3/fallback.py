from __future__ import annotations

from .utils import top1


def select_by_cluster_support(pi: dict[int | str, float], cluster_support: dict[int, int]) -> int:
    ranked = sorted(
        [(a, p) for a, p in pi.items() if a != "OTHER"],
        key=lambda kv: (cluster_support.get(kv[0], 0), kv[1]),
        reverse=True,
    )
    if ranked:
        return int(ranked[0][0])
    return 0


def fallback_logic(meta: dict, diagnostics: dict, ver_results: list, rng) -> int:
    ft = diagnostics.get("failure_type_top", "")
    pi = diagnostics.get("pi", {})
    cluster_support = diagnostics.get("cluster_support", {})

    if ft == "misparse":
        # Conservative fallback after reparsing is to trust cluster diversity.
        return select_by_cluster_support(pi, cluster_support)
    if ft == "geometry_blind":
        return select_by_cluster_support(pi, cluster_support)
    if ft == "missing_case":
        return select_by_cluster_support(pi, cluster_support)
    if ft == "tool_unstable":
        return select_by_cluster_support(pi, cluster_support)

    a1, _ = top1(pi)
    if a1 == "OTHER":
        return 0
    return int(a1)


def should_fallback(answer: int, diagnostics: dict) -> bool:
    frag = diagnostics.get("fragility", {}).get(answer, 1.0)
    ft = diagnostics.get("failure_type_top", "")
    return bool(frag > 0.8 or ft in {"misparse", "geometry_blind", "missing_case", "tool_unstable"})

