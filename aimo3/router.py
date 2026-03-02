from __future__ import annotations


RISK_PROFILE_MAP = {
    "number_theory": "modular_misstep",
    "geometry": "formalization_blindspot",
    "combinatorics": "casework_overcount",
    "algebra": "symbolic_mismatch",
    "mixed": "parser_risk",
}


def route(meta: dict, latex: str) -> dict:
    domain = meta.get("domain", "mixed")
    difficulty = float(meta.get("difficulty", 0.5))
    tool_need = domain in {"number_theory", "algebra", "combinatorics", "geometry"}
    if difficulty < 0.35:
        level = "easy"
    elif difficulty < 0.75:
        level = "medium"
    else:
        level = "hard"

    return {
        "domain": domain,
        "difficulty": difficulty,
        "difficulty_bucket": level,
        "tool_need": tool_need,
        "risk_profile": RISK_PROFILE_MAP.get(domain, "general"),
        "try_symbolic_first": domain in {"number_theory", "algebra"},
        "use_backsolve": difficulty >= 0.5,
        "allow_hard_mode": difficulty >= 0.7,
    }

