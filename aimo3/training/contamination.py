from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable

from aimo3.parsing import normalize_latex

_BLOCKLIST_PATTERNS = [
    re.compile(r"\baimo3\b", re.IGNORECASE),
    re.compile(r"reference\.csv", re.IGNORECASE),
    re.compile(r"kaggle discussion", re.IGNORECASE),
]


def normalized_hash(text: str) -> str:
    normalized = normalize_latex(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def blocked_source(text: str, source: str = "") -> bool:
    haystack = f"{source}\n{text}"
    return any(p.search(haystack) for p in _BLOCKLIST_PATTERNS)


def shingles(text: str, k: int = 5) -> set[str]:
    tokens = normalize_latex(text).lower().split()
    if len(tokens) < k:
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[i : i + k]) for i in range(len(tokens) - k + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def drop_near_duplicates(
    records: Iterable[dict],
    text_key: str = "prompt",
    threshold: float = 0.85,
) -> list[dict]:
    kept: list[dict] = []
    cached_shingles: list[set[str]] = []
    for row in records:
        text = str(row.get(text_key, ""))
        sig = shingles(text)
        duplicate = any(jaccard(sig, existing) >= threshold for existing in cached_shingles)
        if duplicate:
            continue
        cached_shingles.append(sig)
        kept.append(row)
    return kept
