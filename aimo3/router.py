from __future__ import annotations

from collections import Counter

from aimo3.config import SolverConfig
from aimo3.models import Difficulty, Domain, ProblemMetadata, RouteDecision


_DOMAIN_KEYWORDS: dict[Domain, tuple[str, ...]] = {
    Domain.GEOMETRY: ("triangle", "circle", "angle", "cyclic", "circumcircle", "incircle"),
    Domain.NUMBER_THEORY: ("mod", "remainder", "divides", "prime", "gcd", "lcm"),
    Domain.COMBINATORICS: ("count", "ways", "number of", "tournament", "arrange"),
    Domain.ALGEBRA: ("function", "equation", "roots", "polynomial", "sum"),
}


def _pick_domain(meta: ProblemMetadata) -> Domain:
    text = meta.normalized_text.lower()
    counter: Counter[Domain] = Counter()
    for domain, keywords in _DOMAIN_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text:
                counter[domain] += 1
    for hint in meta.detected_domain_hints:
        for domain in Domain:
            if domain.value == hint:
                counter[domain] += 1
    if not counter:
        return Domain.MIXED
    domain, count = counter.most_common(1)[0]
    if count <= 1 and len(counter) > 1:
        return Domain.MIXED
    return domain


def _difficulty_score(meta: ProblemMetadata) -> float:
    text = meta.normalized_text.lower()
    words = len(text.split())
    eq_count = len(meta.extracted_equations)
    number_count = len(meta.numbers)
    hard_markers = sum(
        marker in text
        for marker in (
            "for all sufficiently large",
            "largest possible",
            "unique",
            "across all",
            "let sequence",
            "what is the remainder when",
        )
    )
    score = 0.0
    score += min(words / 450.0, 0.5)
    score += min(eq_count / 6.0, 0.2)
    score += min(number_count / 40.0, 0.15)
    score += min(hard_markers * 0.08, 0.3)
    return min(score, 1.0)


def route_problem(meta: ProblemMetadata, config: SolverConfig) -> RouteDecision:
    domain = _pick_domain(meta)
    score = _difficulty_score(meta)
    if score < 0.35:
        difficulty = Difficulty.EASY
    elif score < 0.7:
        difficulty = Difficulty.MEDIUM
    else:
        difficulty = Difficulty.HARD

    requires_tool = domain in {Domain.NUMBER_THEORY, Domain.COMBINATORICS, Domain.MIXED}
    if meta.modulus is not None:
        requires_tool = True

    use_backsolve = difficulty != Difficulty.EASY or domain in {Domain.GEOMETRY, Domain.MIXED}
    allow_hard_mode = config.enable_hard_mode and difficulty == Difficulty.HARD
    try_symbolic_first = domain != Domain.GEOMETRY

    if difficulty == Difficulty.EASY:
        n_tool, n_cot, n_backsolve = 1, 1, 0
    elif difficulty == Difficulty.MEDIUM:
        n_tool, n_cot, n_backsolve = 2, 2, 1
    else:
        n_tool, n_cot, n_backsolve = 4, 3, 2

    return RouteDecision(
        domain=domain,
        difficulty=difficulty,
        difficulty_score=score,
        requires_tool=requires_tool,
        try_symbolic_first=try_symbolic_first,
        use_backsolve=use_backsolve,
        allow_hard_mode=allow_hard_mode,
        n_tool_attempts=n_tool,
        n_cot_attempts=n_cot,
        n_backsolve_attempts=n_backsolve,
    )
