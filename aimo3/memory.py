from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from aimo3.models import ProblemMetadata
from aimo3.parsing import normalize_latex


def _shingles(text: str, k: int = 4) -> set[str]:
    tokens = normalize_latex(text).lower().split()
    if len(tokens) < k:
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[i : i + k]) for i in range(len(tokens) - k + 1)}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


@dataclass(frozen=True)
class MemoryMatch:
    answer: int
    similarity: float
    source_id: str
    source_problem: str


@dataclass
class _MemoryItem:
    pid: str
    problem: str
    answer: int
    norm_text: str
    shingles: set[str]


class MemoryRetriever:
    def __init__(self, reference_path: Path):
        self.reference_path = reference_path
        self._items: list[_MemoryItem] = []
        self._hash_index: dict[str, _MemoryItem] = {}
        self._load()

    def _load(self) -> None:
        if not self.reference_path.exists():
            return
        with self.reference_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                problem = str(row.get("problem", "")).strip()
                answer_raw = str(row.get("answer", "")).strip()
                pid = str(row.get("id", "")).strip()
                if not problem or not answer_raw:
                    continue
                try:
                    answer = int(answer_raw)
                except ValueError:
                    continue
                norm_text = normalize_latex(problem)
                item = _MemoryItem(
                    pid=pid,
                    problem=problem,
                    answer=answer,
                    norm_text=norm_text,
                    shingles=_shingles(norm_text),
                )
                self._items.append(item)
                self._hash_index[item.norm_text] = item

    def lookup(self, meta: ProblemMetadata, threshold: float) -> MemoryMatch | None:
        exact = self._hash_index.get(meta.normalized_text)
        if exact is not None:
            return MemoryMatch(
                answer=exact.answer,
                similarity=1.0,
                source_id=exact.pid,
                source_problem=exact.problem,
            )

        if not self._items:
            return None
        target = _shingles(meta.normalized_text)
        best_item: _MemoryItem | None = None
        best_score = 0.0
        for item in self._items:
            score = _jaccard(target, item.shingles)
            if score > best_score:
                best_score = score
                best_item = item
        if best_item is None or best_score < threshold:
            return None
        return MemoryMatch(
            answer=best_item.answer,
            similarity=best_score,
            source_id=best_item.pid,
            source_problem=best_item.problem,
        )
