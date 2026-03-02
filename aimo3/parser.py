from __future__ import annotations

import re
from dataclasses import dataclass

from .utils import hash64, tokens


_MOD_PATTERNS = [
    re.compile(r"mod(?:ulo)?\s*(\d{1,9})", re.IGNORECASE),
    re.compile(r"remainder\s+when\s+divided\s+by\s+(\d{1,9})", re.IGNORECASE),
    re.compile(r"\\bmod\s*(\d{1,9})", re.IGNORECASE),
]

_EQUATION_PATTERN = re.compile(r"[^\n]{0,120}=[^\n]{0,120}")


@dataclass(slots=True)
class ParseOutput:
    meta: dict
    constraints: dict


def normalize_latex(text: str) -> str:
    replacements = {
        r"\\cdot": "*",
        r"\\times": "*",
        r"\\frac": "frac",
        "\n": " ",
        "\t": " ",
    }
    out = text
    for src, dst in replacements.items():
        out = out.replace(src, dst)
    out = re.sub(r"\s+", " ", out).strip()
    return out


def detect_domain(normalized: str) -> str:
    t = set(tokens(normalized))
    nt_keys = {"mod", "gcd", "lcm", "prime", "divisible", "remainder", "crt"}
    geo_keys = {"triangle", "circle", "angle", "perpendicular", "parallel", "line", "point"}
    comb_keys = {"count", "ways", "permutation", "combination", "subset", "graph"}
    alg_keys = {"polynomial", "equation", "roots", "function", "identity", "symmetric"}

    scores = {
        "number_theory": len(t & nt_keys),
        "geometry": len(t & geo_keys),
        "combinatorics": len(t & comb_keys),
        "algebra": len(t & alg_keys),
    }
    best_domain, best_score = max(scores.items(), key=lambda kv: kv[1])
    if best_score == 0:
        return "mixed"
    return best_domain


def estimate_difficulty(normalized: str, constraints: dict) -> float:
    length_score = min(len(normalized) / 900.0, 1.0)
    nested_defs = normalized.count("let") + normalized.count("suppose")
    nested_score = min(nested_defs / 8.0, 1.0)
    eqn_score = min(len(constraints.get("equations", [])) / 6.0, 1.0)
    return min(1.0, 0.35 * length_score + 0.35 * nested_score + 0.30 * eqn_score)


def extract_modulus(text: str) -> int | None:
    for p in _MOD_PATTERNS:
        m = p.search(text)
        if m:
            try:
                mod = int(m.group(1))
                if mod > 0:
                    return mod
            except ValueError:
                continue
    return None


def extract_equations(text: str) -> list[str]:
    eqs = []
    for m in _EQUATION_PATTERN.finditer(text):
        v = m.group(0).strip()
        if len(v) >= 3:
            eqs.append(v)
    return eqs[:30]


def parse(latex: str, strictness: str = "normal") -> tuple[dict, dict]:
    normalized = normalize_latex(latex)
    modulus = extract_modulus(normalized)
    equations = extract_equations(normalized)
    answer_is_remainder = bool(
        re.search(r"\bremainder\b", normalized, flags=re.IGNORECASE)
        or re.search(r"\bmod(?:ulo)?\b", normalized, flags=re.IGNORECASE)
    )

    constraints = {
        "modulus": modulus,
        "equations": equations,
        "range": (0, 99999),
        "strictness": strictness,
        "answer_should_be_remainder": answer_is_remainder,
        "num_constraints_extracted": (1 if modulus is not None else 0) + len(equations),
    }

    domain = detect_domain(normalized)
    difficulty = estimate_difficulty(normalized, constraints)

    if strictness == "strict":
        constraints["strict_parse_checks"] = {
            "has_modulus": modulus is not None,
            "has_equations": bool(equations),
            "raw_len": len(normalized),
        }

    meta = {
        "raw_latex": latex,
        "normalized": normalized,
        "domain": domain,
        "difficulty": difficulty,
        "problem_hash": str(hash64(normalized)),
        "modulus": modulus,
        "num_constraints_extracted": constraints["num_constraints_extracted"],
        "answer_should_be_remainder": answer_is_remainder,
        "entities": infer_entities(normalized),
        "geometry_indicator": int(domain == "geometry"),
    }
    return meta, constraints


def infer_entities(normalized: str) -> dict:
    vars_found = sorted(set(re.findall(r"\b[a-zA-Z]\b", normalized)))
    return {
        "variables": vars_found[:40],
        "has_diagram_words": any(k in normalized for k in ["triangle", "circle", "angle", "point"]),
        "has_counting_words": any(k in normalized for k in ["count", "ways", "subset", "arrangement"]),
    }
