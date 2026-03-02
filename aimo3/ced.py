from __future__ import annotations

import itertools
from collections import defaultdict

from .types import Candidate
from .utils import jaccard, sigmoid, tokens


def trace_lemma_set(trace_summary: str) -> set[str]:
    keywords = {
        "lte",
        "crt",
        "invariant",
        "contradiction",
        "symmetry",
        "coordinate",
        "complex",
        "parity",
        "mod",
        "bijection",
        "recurrence",
    }
    ts = set(tokens(trace_summary))
    return ts & keywords


def pair_features(ci: Candidate, cj: Candidate) -> dict[str, float]:
    ta = set(tokens(ci.trace_summary))
    tb = set(tokens(cj.trace_summary))
    trace_sim = jaccard(ta, tb)

    lemma_sim = jaccard(trace_lemma_set(ci.trace_summary), trace_lemma_set(cj.trace_summary))

    code_a = set(tokens(ci.tool_code or ""))
    code_b = set(tokens(cj.tool_code or ""))
    code_sim = jaccard(code_a, code_b) if (code_a or code_b) else 0.0

    mode_sim = 1.0 if ci.policy_id == cj.policy_id else 0.0
    path_sim = 1.0 if ci.path_type == cj.path_type else 0.0
    ans_same = 1.0 if ci.answer == cj.answer else 0.0

    return {
        "trace_sim": trace_sim,
        "lemma_sim": lemma_sim,
        "code_sim": code_sim,
        "mode_sim": mode_sim,
        "path_sim": path_sim,
        "ans_same": ans_same,
    }


def rho_from_features(feat: dict[str, float]) -> float:
    z = 0.0
    z += 1.1 * feat["trace_sim"]
    z += 0.9 * feat["lemma_sim"]
    z += 0.8 * feat["code_sim"]
    z += 0.6 * feat["mode_sim"]
    z += 0.5 * feat["path_sim"]
    z += 0.4 * feat["ans_same"]
    z -= 1.4  # bias to avoid over-clustering
    return sigmoid(z)


def compute_rho_matrix(cands: list[Candidate]) -> list[list[float]]:
    n = len(cands)
    rho = [[0.0 for _ in range(n)] for _ in range(n)]
    for i, j in itertools.combinations(range(n), 2):
        feat = pair_features(cands[i], cands[j])
        r = rho_from_features(feat)
        rho[i][j] = r
        rho[j][i] = r
    return rho


def cluster_by_threshold(cands: list[Candidate], rho: list[list[float]], tau: float = 0.7) -> dict[int, list[int]]:
    parent = list(range(len(cands)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(len(cands)):
        for j in range(i + 1, len(cands)):
            if rho[i][j] >= tau:
                union(i, j)

    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(len(cands)):
        groups[find(i)].append(i)

    # normalize cluster ids
    out: dict[int, list[int]] = {}
    for k, (_, members) in enumerate(groups.items()):
        out[k] = members
    return out


def redundancy_weight(i: int, members: list[int], rho: list[list[float]]) -> float:
    denom = 1.0
    for j in members:
        if j != i:
            denom += rho[i][j]
    return 1.0 / max(denom, 1.0)


def assign_clusters_and_weights(cands: list[Candidate]) -> tuple[list[list[float]], dict[int, list[int]], dict[int, float]]:
    if not cands:
        return [], {}, {}
    rho = compute_rho_matrix(cands)
    clusters = cluster_by_threshold(cands, rho, tau=0.7)

    idx_to_cluster: dict[int, int] = {}
    for cid, members in clusters.items():
        for i in members:
            idx_to_cluster[i] = cid

    weights: dict[int, float] = {}
    for i, cand in enumerate(cands):
        cid = idx_to_cluster[i]
        members = clusters[cid]
        cand.cluster_id = cid
        w = redundancy_weight(i, members, rho)
        cand.redundancy_w = w
        weights[i] = w
    return rho, clusters, weights


def answer_cluster_support(cands: list[Candidate], clusters: dict[int, list[int]]) -> dict[int, int]:
    support: dict[int, set[int]] = defaultdict(set)
    for cid, members in clusters.items():
        for idx in members:
            support[cands[idx].answer].add(cid)
    return {ans: len(ids) for ans, ids in support.items()}

