"""Phase 9.1.4 — qa.yaml shape validation.

Cheap parse test: the shipped catalog must always load. Catches typos
in qa.yaml at the unit-test layer instead of failing in CI."""

from __future__ import annotations

from pathlib import Path

import pytest

from evals.regression.cases import load_cases


CATALOG = Path(__file__).resolve().parent.parent / "evals" / "regression" / "qa.yaml"


@pytest.mark.unit
def test_catalog_parses() -> None:
    cases = load_cases(CATALOG)
    assert len(cases) > 0


@pytest.mark.unit
def test_catalog_has_no_duplicate_ids() -> None:
    """load_cases enforces this; explicit assertion documents the contract."""
    cases = load_cases(CATALOG)
    ids = [c.id for c in cases]
    assert len(ids) == len(set(ids))


@pytest.mark.unit
def test_catalog_covers_core_categories() -> None:
    """The starter catalog should hit every category the v4 plan
    enumerates — preference, supersession, abstention, multi-session,
    temporal window — so a regression in any track gets caught."""
    cases = load_cases(CATALOG)
    cats = {c.category for c in cases}
    expected = {
        "preference", "supersession",
        "abstention", "multi_session", "temporal_window",
    }
    missing = expected - cats
    assert not missing, f"catalog missing categories: {missing}"


@pytest.mark.unit
def test_every_case_has_query_and_at_least_one_assertion() -> None:
    """A case with no must_contain / must_not_contain / abstain_required
    is degenerate — it would always pass. Guard against accidentally
    shipping such a case."""
    cases = load_cases(CATALOG)
    for c in cases:
        has_assertion = (
            c.must_contain or c.must_not_contain
            or c.abstain_required or c.abstain_ok
        )
        assert has_assertion, f"case {c.id!r} has no assertion"
