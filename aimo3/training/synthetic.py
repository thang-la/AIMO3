from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class SyntheticSample:
    prompt: str
    answer: int
    domain: str
    family: str
    modulus: int

    def to_sft_record(self) -> dict:
        return {
            "prompt": self.prompt,
            "response": {
                "domain": self.domain,
                "plan": ["extract constraints", "derive expression", "compute integer answer"],
                "tool_calls": [{"language": "python", "code": "ANSWER = ..."}],
                "final": {"answer": int(self.answer), "modulus": int(self.modulus)},
            },
            "meta": {"family": self.family},
        }


def _family_number_theory(rng: random.Random) -> SyntheticSample:
    a = rng.randint(1000, 50000)
    b = rng.randint(1000, 50000)
    m = rng.choice([99991, 100000, 10007])
    prompt = (
        f"Let N = {a}^2 + {b}^2 + {a}*{b}. "
        f"What is the remainder when N is divided by {m}?"
    )
    answer = (a * a + b * b + a * b) % m
    return SyntheticSample(prompt=prompt, answer=answer, domain="number_theory", family="nt_mod", modulus=m)


def _family_algebra(rng: random.Random) -> SyntheticSample:
    x = rng.randint(20, 200)
    y = rng.randint(20, 200)
    m = 100000
    s = x + y
    p = x * y
    prompt = (
        f"Positive integers x,y satisfy x+y={s} and xy={p}. "
        "Find the remainder when x^3+y^3 is divided by 100000."
    )
    answer = (x**3 + y**3) % m
    return SyntheticSample(prompt=prompt, answer=answer, domain="algebra", family="alg_sym", modulus=m)


def _family_combinatorics(rng: random.Random) -> SyntheticSample:
    n = rng.randint(8, 24)
    k = rng.randint(3, n - 2)
    m = 99991
    prompt = (
        f"How many subsets of size {k} can be chosen from a set of size {n}? "
        f"Return the remainder modulo {m}."
    )
    # Multiplicative nCk
    numer = 1
    denom = 1
    for i in range(1, k + 1):
        numer *= n - i + 1
        denom *= i
    answer = (numer // denom) % m
    return SyntheticSample(
        prompt=prompt,
        answer=answer,
        domain="combinatorics",
        family="comb_choose",
        modulus=m,
    )


def _family_geometry(rng: random.Random) -> SyntheticSample:
    a = rng.randint(10, 300)
    b = rng.randint(10, 300)
    c = rng.randint(abs(a - b) + 1, a + b - 1)
    m = 100000
    perimeter = a + b + c
    prompt = (
        f"A triangle has integer side lengths a={a}, b={b}, c={c}. "
        "Find the remainder when (a+b+c)^2 + abc is divided by 100000."
    )
    answer = (perimeter * perimeter + a * b * c) % m
    return SyntheticSample(prompt=prompt, answer=answer, domain="geometry", family="geo_coord", modulus=m)


def generate_synthetic_dataset(count: int, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    families = [_family_number_theory, _family_algebra, _family_combinatorics, _family_geometry]
    records: list[dict] = []
    for _ in range(count):
        family_fn = rng.choice(families)
        sample = family_fn(rng)
        records.append(sample.to_sft_record())
    return records
