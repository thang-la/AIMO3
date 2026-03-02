from __future__ import annotations

from aimo3.parser import parse
from aimo3.types import Candidate
from aimo3.verifier import verify_shallow


def test_parse_sets_stable_metadata_fields() -> None:
    meta1, constraints1 = parse("Find the remainder when 1234 is divided by 97.")
    meta2, constraints2 = parse("Find the remainder when 1234 is divided by 97.")

    assert meta1["problem_hash"] == meta2["problem_hash"]
    assert meta1["num_constraints_extracted"] == constraints1["num_constraints_extracted"]
    assert constraints1["answer_should_be_remainder"] is True
    assert constraints2["answer_should_be_remainder"] is True


def test_verify_shallow_remainder_constraint() -> None:
    meta, constraints = parse("Find the remainder when n is divided by 7.")
    ok = Candidate(id=1, answer=3, policy_id="A_SYMBOLIC", path_type="P0", trace="", trace_summary="")
    bad = Candidate(id=2, answer=10, policy_id="A_SYMBOLIC", path_type="P0", trace="", trace_summary="")

    assert verify_shallow(ok, meta, constraints).hard_ok is True
    assert verify_shallow(bad, meta, constraints).hard_ok is False

