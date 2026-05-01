"""Unit tests for the bench trend.csv accumulator + reader."""

from __future__ import annotations

import csv
from types import SimpleNamespace

import pytest

from scripts.bench.trend import (
    TREND_HEADERS,
    _features_from_stack,
    append_trend_row,
    format_trend_markdown,
    read_trend,
)


# ── append_trend_row ──────────────────────────────────────────────────


@pytest.fixture
def summary_dict() -> dict:
    return {
        "primary_metric": 87.5,
        "primary_metric_name": "accuracy_pct",
        "total": 78,
        "metadata": {
            "answer_model": "openai/gpt-5.4-mini",
            "judges": ["openai/gpt-5.5", "anthropic/claude-sonnet-4-6"],
        },
    }


@pytest.mark.unit
def test_append_creates_file_with_headers(tmp_path, summary_dict) -> None:
    p = tmp_path / "trend.csv"
    append_trend_row(
        p,
        summary=summary_dict,
        variant="s",
        category="knowledge-update",
        features=["multi_query"],
        git_sha="abc1234",
    )
    text = p.read_text().splitlines()
    assert text[0].startswith("timestamp,")  # header line
    # All schema columns present in the header
    for h in TREND_HEADERS:
        assert h in text[0]
    # One data row
    assert len(text) == 2


@pytest.mark.unit
def test_append_subsequent_rows_skip_headers(tmp_path, summary_dict) -> None:
    p = tmp_path / "trend.csv"
    for _ in range(3):
        append_trend_row(
            p,
            summary=summary_dict,
            variant="s",
            category="knowledge-update",
            features=["hyde"],
            git_sha="abc1234",
        )
    rows = list(csv.DictReader(p.open()))
    assert len(rows) == 3, f"expected 3 data rows, got {len(rows)}"
    # All same input → all rows have same fingerprint
    assert all(r["score_pct"] == "87.5" for r in rows)


@pytest.mark.unit
def test_append_records_features_and_metadata(tmp_path, summary_dict) -> None:
    p = tmp_path / "trend.csv"
    row = append_trend_row(
        p,
        summary=summary_dict,
        variant="s",
        category="multi-session",
        features=["multi_query", "hyde", "self_consistency"],
        git_sha="deadbee",
        run_label="ablation_a",
    )
    assert row.features == "multi_query,hyde,self_consistency"
    assert row.git_sha == "deadbee"
    assert row.run_label == "ablation_a"
    assert row.judges == "openai/gpt-5.5;anthropic/claude-sonnet-4-6"
    assert row.score_pct == 87.5
    assert row.n == 78


@pytest.mark.unit
def test_append_handles_empty_features_list(tmp_path, summary_dict) -> None:
    p = tmp_path / "trend.csv"
    row = append_trend_row(
        p, summary=summary_dict, variant="s", category=None,
        features=[], git_sha="x",
    )
    assert row.features == ""
    assert row.category == "all"  # None → "all" canonicalization


@pytest.mark.unit
def test_append_creates_parent_dir(tmp_path, summary_dict) -> None:
    p = tmp_path / "nested" / "deeper" / "trend.csv"
    append_trend_row(
        p, summary=summary_dict, variant="s", category="x",
        features=[], git_sha="x",
    )
    assert p.exists()


# ── read_trend ────────────────────────────────────────────────────────


@pytest.mark.unit
def test_read_trend_empty_file_returns_empty(tmp_path) -> None:
    assert read_trend(tmp_path / "missing.csv") == []


@pytest.mark.unit
def test_read_trend_round_trip(tmp_path, summary_dict) -> None:
    p = tmp_path / "trend.csv"
    append_trend_row(
        p, summary=summary_dict, variant="s", category="x",
        features=["multi_query"], git_sha="abc",
    )
    rows = read_trend(p)
    assert len(rows) == 1
    assert rows[0]["variant"] == "s"
    assert rows[0]["category"] == "x"
    assert rows[0]["features"] == "multi_query"


# ── format_trend_markdown ─────────────────────────────────────────────


@pytest.mark.unit
def test_format_trend_empty_returns_placeholder() -> None:
    md = format_trend_markdown([])
    assert "No trend rows yet" in md


@pytest.mark.unit
def test_format_trend_renders_table_headers() -> None:
    rows = [{
        "timestamp": "2026-04-29T14:00:00+00:00",
        "date": "20260429",
        "git_sha": "abc",
        "variant": "s",
        "category": "knowledge-update",
        "n": "78",
        "score_pct": "87.5",
        "answer_model": "x",
        "judges": "j1;j2",
        "features": "multi_query",
        "run_label": "smoke",
    }]
    md = format_trend_markdown(rows)
    assert "| Variant | Category | Date" in md
    assert "knowledge-update" in md
    assert "87.5%" in md


@pytest.mark.unit
def test_format_trend_computes_deltas_within_group() -> None:
    """Two rows in the same (variant, category) → second shows delta
    against the first."""
    rows = [
        {
            "timestamp": "2026-04-29T10:00:00+00:00",
            "date": "20260429",
            "git_sha": "a",
            "variant": "s",
            "category": "knowledge-update",
            "n": "78",
            "score_pct": "80.0",
            "answer_model": "x",
            "judges": "j",
            "features": "",
            "run_label": "",
        },
        {
            "timestamp": "2026-04-29T11:00:00+00:00",
            "date": "20260429",
            "git_sha": "b",
            "variant": "s",
            "category": "knowledge-update",
            "n": "78",
            "score_pct": "88.0",
            "answer_model": "x",
            "judges": "j",
            "features": "multi_query",
            "run_label": "",
        },
    ]
    md = format_trend_markdown(rows)
    # First row has no delta (no previous in group)
    # Second row shows +8.0
    assert "+8.0" in md


@pytest.mark.unit
def test_format_trend_groups_by_variant_and_category() -> None:
    """Different (variant, category) groups don't share the delta lineage."""
    rows = [
        {
            "timestamp": "2026-04-29T10:00:00+00:00",
            "date": "20260429", "git_sha": "a",
            "variant": "s", "category": "X",
            "n": "10", "score_pct": "80.0",
            "answer_model": "x", "judges": "j",
            "features": "", "run_label": "",
        },
        {
            "timestamp": "2026-04-29T11:00:00+00:00",
            "date": "20260429", "git_sha": "b",
            "variant": "s", "category": "Y",
            "n": "10", "score_pct": "90.0",
            "answer_model": "x", "judges": "j",
            "features": "", "run_label": "",
        },
    ]
    md = format_trend_markdown(rows)
    # Y row should NOT have a delta because it's in a different group
    # from X. Verify by counting "+" or "-" delta markers — there should
    # be 0 (each is the first in its group).
    assert "+10.0" not in md
    assert "-10.0" not in md


# ── _features_from_stack ──────────────────────────────────────────────


@pytest.mark.unit
def test_features_from_stack_default_off() -> None:
    """A default StackConfig has all feature flags off → empty list."""
    from attestor.config import (
        HydeCfg, MultiQueryCfg, RetrievalCfg,
        SelfConsistencyCfg, CritiqueReviseCfg, TemporalPrefilterCfg,
    )

    stack = SimpleNamespace(
        retrieval=RetrievalCfg(
            multi_query=MultiQueryCfg(enabled=False),
            temporal_prefilter=TemporalPrefilterCfg(enabled=False),
            hyde=HydeCfg(enabled=False),
        ),
        self_consistency=SelfConsistencyCfg(enabled=False),
        critique_revise=CritiqueReviseCfg(enabled=False),
    )
    assert _features_from_stack(stack) == []


@pytest.mark.unit
def test_features_from_stack_all_on() -> None:
    """Every flag flipped on → all 5 names appear in feature list."""
    from attestor.config import (
        HydeCfg, MultiQueryCfg, RetrievalCfg,
        SelfConsistencyCfg, CritiqueReviseCfg, TemporalPrefilterCfg,
    )

    stack = SimpleNamespace(
        retrieval=RetrievalCfg(
            multi_query=MultiQueryCfg(enabled=True),
            temporal_prefilter=TemporalPrefilterCfg(enabled=True),
            hyde=HydeCfg(enabled=True),
        ),
        self_consistency=SelfConsistencyCfg(enabled=True),
        critique_revise=CritiqueReviseCfg(enabled=True),
    )
    features = _features_from_stack(stack)
    assert set(features) == {
        "multi_query", "hyde", "temporal_prefilter",
        "self_consistency", "critique_revise",
    }


@pytest.mark.unit
def test_features_from_stack_mixed() -> None:
    from attestor.config import (
        HydeCfg, MultiQueryCfg, RetrievalCfg,
        SelfConsistencyCfg, CritiqueReviseCfg, TemporalPrefilterCfg,
    )

    stack = SimpleNamespace(
        retrieval=RetrievalCfg(
            multi_query=MultiQueryCfg(enabled=True),
            temporal_prefilter=TemporalPrefilterCfg(enabled=False),
            hyde=HydeCfg(enabled=False),
        ),
        self_consistency=SelfConsistencyCfg(enabled=True),
        critique_revise=CritiqueReviseCfg(enabled=False),
    )
    features = _features_from_stack(stack)
    assert features == ["multi_query", "self_consistency"]
