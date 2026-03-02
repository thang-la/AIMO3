from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from typing import Iterable


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def softmax_dict(scores: dict[int | str, float]) -> dict[int | str, float]:
    if not scores:
        return {}
    m = max(scores.values())
    exps = {k: math.exp(v - m) for k, v in scores.items()}
    z = sum(exps.values())
    if z == 0:
        return {k: 1.0 / len(scores) for k in scores}
    return {k: v / z for k, v in exps.items()}


def normalize_dict(values: dict[int | str, float]) -> dict[int | str, float]:
    s = sum(values.values())
    if s <= 0:
        return {k: 1.0 / len(values) for k in values}
    return {k: v / s for k, v in values.items()}


def topk_items(d: dict[int | str, float], k: int = 2) -> list[tuple[int | str, float]]:
    return sorted(d.items(), key=lambda kv: kv[1], reverse=True)[:k]


def top1(d: dict[int | str, float]) -> tuple[int | str, float]:
    if not d:
        return "OTHER", 1.0
    return max(d.items(), key=lambda kv: kv[1])


def top2(d: dict[int | str, float]) -> tuple[tuple[int | str, float], tuple[int | str, float]]:
    items = topk_items(d, k=2)
    if len(items) == 1:
        return items[0], ("OTHER", 0.0)
    return items[0], items[1]


def safe_entropy(probs: Iterable[float]) -> float:
    h = 0.0
    for p in probs:
        if p > 0:
            h -= p * math.log(p)
    return h


def hash32(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:4], "big")


def hash64(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def tokens(text: str) -> list[str]:
    return [t for t in "".join(ch.lower() if ch.isalnum() else " " for ch in text).split() if t]


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    u = a | b
    if not u:
        return 0.0
    return len(a & b) / len(u)


def group_by_answer(items: list[tuple[int, float]]) -> dict[int, float]:
    out: dict[int, float] = defaultdict(float)
    for ans, val in items:
        out[ans] += val
    return dict(out)

