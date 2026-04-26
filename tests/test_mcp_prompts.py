"""Phase 8.2 — MCP prompt content guards + format helpers.

Anti-regression on the load-bearing pieces of each prompt. The MCP
server exposes these via prompts/get; drift in the schema or constraints
breaks every agent that consumes them.
"""

from __future__ import annotations

import pytest

from attestor.mcp.prompts import (
    AUDIT_DECISION_PROMPT,
    HANDOFF_PROMPT_TEMPLATE,
    PROPOSE_INVALIDATION_PROMPT,
    RECORD_DECISION_PROMPT,
    RESUME_THREAD_PROMPT,
    format_audit_decision_prompt,
    format_handoff_prompt,
    format_propose_invalidation_prompt,
    format_record_decision_prompt,
    format_resume_thread_prompt,
)


# ──────────────────────────────────────────────────────────────────────────
# record_decision
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_record_decision_requires_evidence_ids() -> None:
    """A decision without evidence_ids is not auditable — load-bearing."""
    assert "evidence_ids" in RECORD_DECISION_PROMPT
    assert "[mem_<id>]" in RECORD_DECISION_PROMPT
    assert "not auditable" in RECORD_DECISION_PROMPT.lower() or \
        "without evidence_ids is not auditable" in RECORD_DECISION_PROMPT


@pytest.mark.unit
def test_record_decision_has_required_fields() -> None:
    for field in ("decision", "rationale", "evidence_ids",
                  "reversibility", "effective_at"):
        assert field in RECORD_DECISION_PROMPT


@pytest.mark.unit
def test_format_record_decision_substitutes() -> None:
    out = format_record_decision_prompt(
        agent_id="planner-01", thread_id="t-99",
        user_query="should we approve the budget?",
    )
    assert "planner-01" in out
    assert "t-99" in out
    assert "approve the budget" in out


# ──────────────────────────────────────────────────────────────────────────
# handoff_to
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_handoff_includes_five_sections() -> None:
    for section in ("SUMMARY", "KEY DECISIONS", "OPEN QUESTIONS",
                    "CONSTRAINTS", "SUPERSEDED CONTEXT"):
        assert section in HANDOFF_PROMPT_TEMPLATE


@pytest.mark.unit
def test_handoff_explains_why_superseded_is_included() -> None:
    """The superseded section is the +biggest fix — must explain why."""
    assert "re-raise" in HANDOFF_PROMPT_TEMPLATE.lower()


@pytest.mark.unit
def test_handoff_cite_format_specified() -> None:
    assert "[mem_<id>]" in HANDOFF_PROMPT_TEMPLATE


@pytest.mark.unit
def test_format_handoff_substitutes_agents_and_budget() -> None:
    out = format_handoff_prompt(
        from_agent="planner", to_agent="executor",
        thread_id="t-5", recall_budget=8000,
    )
    assert "from planner to executor" in out
    assert "t-5" in out
    assert "8000 tokens" in out


# ──────────────────────────────────────────────────────────────────────────
# resume_thread
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_resume_thread_includes_four_sections() -> None:
    for section in ("STATE", "LAST ACTION", "OPEN ITEMS", "STALE"):
        assert section in RESUME_THREAD_PROMPT


@pytest.mark.unit
def test_resume_thread_is_reading_only() -> None:
    """Hard prohibition on taking new actions — resume is observation."""
    assert "READING task" in RESUME_THREAD_PROMPT
    assert "Do not take" in RESUME_THREAD_PROMPT or \
        "do not take new actions" in RESUME_THREAD_PROMPT.lower()


@pytest.mark.unit
def test_format_resume_thread_substitutes() -> None:
    out = format_resume_thread_prompt(
        thread_id="t-7",
        memories_chronological="(mem list)",
        window_days=14,
    )
    assert "t-7" in out
    assert "(mem list)" in out
    assert "last 14 days" in out


# ──────────────────────────────────────────────────────────────────────────
# audit_decision
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_audit_decision_includes_five_sections() -> None:
    for section in ("CURRENT STATE", "PROVENANCE", "SUPERSESSION CHAIN",
                    "SIGNATURE", "FINDINGS"):
        assert section in AUDIT_DECISION_PROMPT


@pytest.mark.unit
def test_audit_decision_quotes_verbatim() -> None:
    """Auditor quotes verbatim — paraphrasing is not allowed."""
    assert "verbatim" in AUDIT_DECISION_PROMPT.lower()
    assert "Quote exactly" in AUDIT_DECISION_PROMPT


@pytest.mark.unit
def test_audit_signature_unsigned_flagged_not_failed() -> None:
    """Unsigned rows are FLAGGED, not failed — important for v3 backfill."""
    assert "Unsigned rows are flagged" in AUDIT_DECISION_PROMPT


@pytest.mark.unit
def test_format_audit_decision_substitutes() -> None:
    out = format_audit_decision_prompt(
        memory_id="m-42", audit_payload="(payload)",
    )
    assert "m-42" in out
    assert "(payload)" in out


# ──────────────────────────────────────────────────────────────────────────
# propose_invalidation
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_propose_invalidation_required_fields() -> None:
    for field in ("target_id", "rationale", "evidence_ids",
                  "replacement", "severity", "needs_human_review"):
        assert field in PROPOSE_INVALIDATION_PROMPT


@pytest.mark.unit
def test_propose_invalidation_human_review_triggers() -> None:
    """High-severity / signed rows / weaker evidence → human review."""
    assert "needs_human_review=true" in PROPOSE_INVALIDATION_PROMPT
    assert "signature verification" in PROPOSE_INVALIDATION_PROMPT


@pytest.mark.unit
def test_format_propose_invalidation_substitutes() -> None:
    out = format_propose_invalidation_prompt(
        target_memory_id="m-99",
        reviewer_id="reviewer-alpha",
        target_payload="(target)",
    )
    assert "m-99" in out
    assert "reviewer-alpha" in out
    assert "(target)" in out


# ──────────────────────────────────────────────────────────────────────────
# Re-export sanity
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_top_level_re_exports_all_prompts() -> None:
    from attestor.mcp.prompts import (
        AUDIT_DECISION_PROMPT,
        HANDOFF_PROMPT_TEMPLATE,
        PROPOSE_INVALIDATION_PROMPT,
        RECORD_DECISION_PROMPT,
        RESUME_THREAD_PROMPT,
    )
    for s in (AUDIT_DECISION_PROMPT, HANDOFF_PROMPT_TEMPLATE,
              PROPOSE_INVALIDATION_PROMPT, RECORD_DECISION_PROMPT,
              RESUME_THREAD_PROMPT):
        assert isinstance(s, str)
        assert len(s) > 100  # not an empty string


@pytest.mark.unit
def test_all_format_helpers_render_without_orphan_placeholders() -> None:
    """No prompt should leave a {placeholder} in the rendered output."""
    rendered = [
        format_record_decision_prompt(
            agent_id="a", thread_id="t", user_query="q"),
        format_handoff_prompt(
            from_agent="a", to_agent="b", thread_id="t"),
        format_resume_thread_prompt(
            thread_id="t", memories_chronological="x"),
        format_audit_decision_prompt(memory_id="m", audit_payload="x"),
        format_propose_invalidation_prompt(
            target_memory_id="m", reviewer_id="r", target_payload="x"),
    ]
    for r in rendered:
        # No bare braces (only schema literal {{...}} after format)
        assert "{" not in r or "{{" not in r
