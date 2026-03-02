from __future__ import annotations

import hashlib
import re

from aimo3.models import ProblemMetadata

_MOD_PATTERNS = [
    re.compile(r"\bmod(?:ulo|ulus)?\s*(\d+)", re.IGNORECASE),
    re.compile(r"remainder when .* divided by\s*(\d+)", re.IGNORECASE),
    re.compile(r"divided by\s*(\d+)", re.IGNORECASE),
]
_POWER_PATTERN = re.compile(r"(\d+)\s*\^\s*\{?(\d+)\}?")
_NUMBER_PATTERN = re.compile(r"\b\d+\b")
_VAR_PATTERN = re.compile(r"\b[a-zA-Z]\b")


def normalize_latex(text: str) -> str:
    out = text
    replacements = {
        "\n": " ",
        "\t": " ",
        "\\cdot": "*",
        "\\times": "*",
        "\\left": "",
        "\\right": "",
        "\\,": " ",
        "$": "",
    }
    for src, dst in replacements.items():
        out = out.replace(src, dst)
    out = re.sub(r"\s+", " ", out).strip()
    return out


def _token_to_int(token: str) -> int | None:
    token = token.strip()
    if token.isdigit():
        return int(token)
    m = _POWER_PATTERN.fullmatch(token)
    if m:
        base = int(m.group(1))
        exp = int(m.group(2))
        if exp > 12:
            return None
        return base**exp
    return None


def extract_modulus(text: str) -> int | None:
    normalized = normalize_latex(text)
    for pattern in _MOD_PATTERNS:
        match = pattern.search(normalized)
        if match:
            return int(match.group(1))

    # Handle forms like "10^{5}" near remainder statements.
    if "remainder" in normalized.lower() and "divided by" in normalized.lower():
        tail = normalized.lower().split("divided by", 1)[1]
        power_match = _POWER_PATTERN.search(tail.replace(" ", ""))
        if power_match:
            return int(power_match.group(1)) ** int(power_match.group(2))
    return None


def extract_equation_like_substrings(text: str) -> list[str]:
    normalized = normalize_latex(text)
    candidates = re.split(r"[.;]", normalized)
    equations = [c.strip() for c in candidates if "=" in c]
    return [e for e in equations if len(e) >= 3]


def extract_numbers(text: str, limit: int = 64) -> list[int]:
    normalized = normalize_latex(text)
    numbers = [int(m.group(0)) for m in _NUMBER_PATTERN.finditer(normalized)]
    return numbers[:limit]


def extract_variables(text: str, limit: int = 64) -> list[str]:
    normalized = normalize_latex(text)
    vars_found = []
    seen = set()
    for match in _VAR_PATTERN.finditer(normalized):
        var = match.group(0)
        if var not in seen:
            vars_found.append(var)
            seen.add(var)
        if len(vars_found) >= limit:
            break
    return vars_found


def detect_domain_hints(text: str) -> list[str]:
    normalized = normalize_latex(text).lower()
    hints = []
    keyword_groups = {
        "geometry": ["triangle", "circle", "angle", "circum", "incircle", "cyclic"],
        "number_theory": ["divides", "mod", "remainder", "prime", "gcd", "lcm"],
        "algebra": ["function", "polynomial", "equation", "roots", "sum"],
        "combinatorics": ["count", "number of", "ways", "tournament", "permutation"],
    }
    for name, words in keyword_groups.items():
        if any(word in normalized for word in words):
            hints.append(name)
    return hints


def parse_problem(pid: str, text: str) -> ProblemMetadata:
    normalized = normalize_latex(text)
    statement_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return ProblemMetadata(
        pid=pid,
        raw_text=text,
        normalized_text=normalized,
        statement_hash=statement_hash,
        modulus=extract_modulus(text),
        extracted_equations=extract_equation_like_substrings(text),
        numbers=extract_numbers(text),
        variables=extract_variables(text),
        detected_domain_hints=detect_domain_hints(text),
    )
