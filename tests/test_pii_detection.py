"""Unit tests for PII detection + opt-in redaction at ingest.

The ``attestor.compliance.pii`` module ships a regex-baseline detector
plus an opt-in LLM-judged path. ``AgentMemory.add()`` runs the detector
when ``stack.compliance.pii.mode`` is anything other than ``off``.

Three modes (mirrored from the YAML block):

    off      — detector never fires; byte-identical to legacy add().
    flag     — findings written to metadata['_pii_findings']; content
               unchanged.
    redact   — content rewritten with [REDACTED:<TYPE>] tokens; original
               content sealed via metadata['_pii_original_sealed'] = True.

These tests cover:

  * ``detect_pii`` regex baseline correctness (email, phone, SSN, credit
    card with Luhn check, IP/URL with embedded auth, DOB / MRN with
    context-keyword gating).
  * ``redact_content`` preserves surrounding text and replaces every
    finding with a typed redaction token.
  * Wire-up into ``AgentMemory.add()``: off / flag / redact behave
    correctly end-to-end, detector errors are isolated, and recall
    surfaces redacted content (NOT the original).
  * False-positive guardrails: bare dates, ordinary 16-digit numbers,
    and date-shaped strings must NOT trip the detector.

The fakes here mirror ``tests/test_contextual_embedding.py`` so the
same injection pattern (registry monkeypatch + dim-check bypass)
applies. RED phase: every test imports a module that does not yet
exist; running ``pytest tests/test_pii_detection.py`` should fail.
"""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import MagicMock

import pytest


# ─────────────────────────────────────────────────────────────────────────
# Module surface
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_module_imports() -> None:
    """The new module + dataclasses + entrypoints should import cleanly."""
    from attestor.compliance.pii import (  # noqa: F401
        PIIFinding,
        PIIResult,
        detect_pii,
        redact_content,
    )


@pytest.mark.unit
def test_pii_finding_is_immutable() -> None:
    """``PIIFinding`` is a frozen dataclass — span + type + confidence stable."""
    from attestor.compliance.pii import PIIFinding

    f = PIIFinding(type="email", span=(0, 10), confidence=1.0, detector="regex_v1")
    with pytest.raises(Exception):
        f.type = "phone"  # type: ignore[misc]


@pytest.mark.unit
def test_pii_cfg_dataclass_defaults() -> None:
    """Defaults: mode=off, llm_model=None, llm_disabled by mode."""
    from attestor.config import PIICfg

    cfg = PIICfg()
    assert cfg.mode == "off"
    assert cfg.llm_model is None


@pytest.mark.unit
def test_yaml_loader_parses_compliance_pii_block(tmp_path) -> None:
    """The YAML loader exposes ``stack.compliance.pii``."""
    from attestor.config import load_stack

    yaml_path = tmp_path / "test_attestor.yaml"
    yaml_path.write_text("""
stack:
  postgres:
    url: postgresql://x/y
  neo4j:
    url: bolt://localhost:7687
    auth: { username: neo4j, password: pw }
    database: neo4j
  embedder:
    provider: voyage
    model: voyage-4
    dimensions: 1024
  models:
    answerer: x
    judge: x
    extraction: x
    distill: x
    verifier: x
    planner: x
    benchmark_default: x
  llm:
    provider: openrouter
  budget: 4000
  parallel: 2
  compliance:
    pii:
      mode: redact
      llm_model: anthropic/claude-haiku-4.5
""")
    stack = load_stack(yaml_path)
    assert stack.compliance.pii.mode == "redact"
    assert stack.compliance.pii.llm_model == "anthropic/claude-haiku-4.5"


@pytest.mark.unit
def test_yaml_loader_defaults_when_block_omitted(tmp_path) -> None:
    """Missing ``stack.compliance`` → default PIICfg(mode='off')."""
    from attestor.config import load_stack

    yaml_path = tmp_path / "test_attestor.yaml"
    yaml_path.write_text("""
stack:
  postgres:
    url: postgresql://x/y
  neo4j:
    url: bolt://localhost:7687
    auth: { username: neo4j, password: pw }
    database: neo4j
  embedder:
    provider: voyage
    model: voyage-4
    dimensions: 1024
  models:
    answerer: x
    judge: x
    extraction: x
    distill: x
    verifier: x
    planner: x
    benchmark_default: x
  llm:
    provider: openrouter
  budget: 4000
  parallel: 2
""")
    stack = load_stack(yaml_path)
    assert stack.compliance.pii.mode == "off"


@pytest.mark.unit
def test_yaml_loader_rejects_unknown_mode(tmp_path) -> None:
    """Unknown mode value should crash hard at load — config is the lever."""
    from attestor.config import load_stack

    yaml_path = tmp_path / "test_attestor.yaml"
    yaml_path.write_text("""
stack:
  postgres: { url: postgresql://x/y }
  neo4j:
    url: bolt://localhost:7687
    auth: { username: neo4j, password: pw }
    database: neo4j
  embedder: { provider: voyage, model: voyage-4, dimensions: 1024 }
  models:
    answerer: x
    judge: x
    extraction: x
    distill: x
    verifier: x
    planner: x
    benchmark_default: x
  llm: { provider: openrouter }
  budget: 4000
  parallel: 2
  compliance:
    pii:
      mode: scrub_with_LLM_oracle    # not one of off|flag|redact
""")
    with pytest.raises(SystemExit):
        load_stack(yaml_path)


# ─────────────────────────────────────────────────────────────────────────
# detect_pii — regex baseline
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_detect_pii_email_basic() -> None:
    """Email regex matches a typical address and returns char span."""
    from attestor.compliance.pii import detect_pii

    text = "contact me at jane@example.com please"
    result = detect_pii(text, mode="flag")
    assert len(result.findings) == 1
    f = result.findings[0]
    assert f.type == "email"
    assert text[f.span[0]:f.span[1]] == "jane@example.com"
    assert f.detector == "regex_v1"
    assert 0.9 <= f.confidence <= 1.0


@pytest.mark.unit
def test_detect_pii_multiple_pii_types() -> None:
    """Three different PII types in one string → three findings."""
    from attestor.compliance.pii import detect_pii

    text = (
        "Email me at jane@example.com or call (415) 555-1234. "
        "My SSN is 123-45-6789 — please keep it secret."
    )
    result = detect_pii(text, mode="flag")
    types = {f.type for f in result.findings}
    assert "email" in types
    assert "phone" in types
    assert "ssn" in types


@pytest.mark.unit
def test_detect_pii_no_findings_returns_empty() -> None:
    """Plain text with no PII → empty findings tuple, redacted_content None."""
    from attestor.compliance.pii import detect_pii

    result = detect_pii("The capital of France is Paris.", mode="flag")
    assert result.findings == ()
    assert result.redacted_content is None


# ─────────────────────────────────────────────────────────────────────────
# Luhn-checked credit card detection
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_luhn_check_credit_card_valid() -> None:
    """A valid Luhn 16-digit number IS flagged as credit_card."""
    from attestor.compliance.pii import detect_pii

    # Visa test number — passes Luhn.
    text = "Card: 4111 1111 1111 1111 expires soon"
    result = detect_pii(text, mode="flag")
    types = {f.type for f in result.findings}
    assert "credit_card" in types


@pytest.mark.unit
def test_luhn_check_credit_card_invalid_random_16_digits() -> None:
    """A random 16-digit string that fails Luhn is NOT flagged as credit_card."""
    from attestor.compliance.pii import detect_pii

    # 1234567890123456 fails Luhn (sum = 70, not multiple of 10).
    text = "order id 1234567890123456 logged"
    result = detect_pii(text, mode="flag")
    types = {f.type for f in result.findings}
    assert "credit_card" not in types


@pytest.mark.unit
def test_luhn_check_credit_card_timestamp_not_flagged() -> None:
    """A 13-digit Unix-millisecond timestamp must not trip the credit card lane."""
    from attestor.compliance.pii import detect_pii

    text = "transaction at 1730473200123 server time"
    result = detect_pii(text, mode="flag")
    types = {f.type for f in result.findings}
    assert "credit_card" not in types


# ─────────────────────────────────────────────────────────────────────────
# Phone false-positive guardrails
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_phone_no_false_positives_on_iso_dates() -> None:
    """ISO date '2024-01-15' must not trip the phone detector."""
    from attestor.compliance.pii import detect_pii

    text = "Filed report on 2024-01-15 and again on 2024-02-20."
    result = detect_pii(text, mode="flag")
    types = {f.type for f in result.findings}
    assert "phone" not in types


@pytest.mark.unit
def test_phone_us_format_detected() -> None:
    """Standard US-format phone is detected as 'phone'."""
    from attestor.compliance.pii import detect_pii

    text = "Call me at (415) 555-1234 anytime."
    result = detect_pii(text, mode="flag")
    types = {f.type for f in result.findings}
    assert "phone" in types


# ─────────────────────────────────────────────────────────────────────────
# DOB context-keyword gating
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_dob_requires_context_keyword_bare_date_skipped() -> None:
    """Bare '1/15/1990' with no birth context → NOT flagged as DOB."""
    from attestor.compliance.pii import detect_pii

    text = "We met on 1/15/1990 in San Francisco."
    result = detect_pii(text, mode="flag")
    types = {f.type for f in result.findings}
    assert "dob" not in types


@pytest.mark.unit
def test_dob_requires_context_keyword_with_keyword_flagged() -> None:
    """'DOB: 1/15/1990' with a birth-context keyword → flagged as DOB."""
    from attestor.compliance.pii import detect_pii

    text = "Patient DOB: 1/15/1990 admitted yesterday."
    result = detect_pii(text, mode="flag")
    types = {f.type for f in result.findings}
    assert "dob" in types


# ─────────────────────────────────────────────────────────────────────────
# URL / IP with auth
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_url_with_auth_detected() -> None:
    """URL with embedded user:pass is type=url_with_auth."""
    from attestor.compliance.pii import detect_pii

    text = "Connect to https://user:pass@host.com/x for the dashboard"
    result = detect_pii(text, mode="flag")
    types = {f.type for f in result.findings}
    assert "url_with_auth" in types


@pytest.mark.unit
def test_url_without_auth_not_flagged_as_url_with_auth() -> None:
    """Plain https URL without embedded creds is NOT flagged as url_with_auth."""
    from attestor.compliance.pii import detect_pii

    text = "See https://example.com/docs for more info."
    result = detect_pii(text, mode="flag")
    types = {f.type for f in result.findings}
    assert "url_with_auth" not in types


# ─────────────────────────────────────────────────────────────────────────
# Redaction
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_redact_mode_email_replaces_with_token() -> None:
    """``mode='redact'`` rewrites email with [REDACTED:EMAIL]."""
    from attestor.compliance.pii import detect_pii

    text = "Email: jane@example.com"
    result = detect_pii(text, mode="redact")
    assert result.redacted_content is not None
    assert "[REDACTED:EMAIL]" in result.redacted_content
    assert "jane@example.com" not in result.redacted_content


@pytest.mark.unit
def test_redact_mode_preserves_structure() -> None:
    """Multiple findings in the same string → all replaced; surrounding
    text intact."""
    from attestor.compliance.pii import detect_pii

    text = (
        "Reach Jane at jane@example.com or (415) 555-1234. "
        "Her SSN is 123-45-6789 do NOT share."
    )
    result = detect_pii(text, mode="redact")
    out = result.redacted_content
    assert out is not None
    # Replacements present
    assert "[REDACTED:EMAIL]" in out
    assert "[REDACTED:PHONE]" in out
    assert "[REDACTED:SSN]" in out
    # Originals fully scrubbed
    assert "jane@example.com" not in out
    assert "(415) 555-1234" not in out
    assert "123-45-6789" not in out
    # Surrounding text preserved
    assert "Reach Jane at" in out
    assert "do NOT share" in out


@pytest.mark.unit
def test_redact_helper_independent_of_detect() -> None:
    """``redact_content`` accepts pre-computed findings and produces tokens."""
    from attestor.compliance.pii import PIIFinding, redact_content

    text = "Email jane@example.com today."
    findings = [
        PIIFinding(
            type="email",
            span=(6, 22),
            confidence=1.0,
            detector="regex_v1",
        ),
    ]
    out = redact_content(text, findings)
    assert "jane@example.com" not in out
    assert "[REDACTED:EMAIL]" in out
    assert out.startswith("Email ")
    assert out.endswith(" today.")


# ─────────────────────────────────────────────────────────────────────────
# Integration with AgentMemory.add() — fakes mirror test_contextual_embedding
# ─────────────────────────────────────────────────────────────────────────


class _FakeDocumentStore:
    """Minimal DocumentStore for AgentMemory.add() integration tests."""

    ROLES = {"document"}

    def __init__(self) -> None:
        self.memories: dict[str, Any] = {}
        self.by_hash: dict[str, Any] = {}

    def insert(self, memory):  # noqa: ANN001
        self.memories[memory.id] = memory
        if memory.content_hash:
            self.by_hash[memory.content_hash] = memory
        return memory

    def get(self, memory_id):  # noqa: ANN001
        return self.memories.get(memory_id)

    def get_by_hash(self, content_hash, namespace="default"):  # noqa: ANN001
        return self.by_hash.get(content_hash)

    def update(self, memory):  # noqa: ANN001
        self.memories[memory.id] = memory

    def list_memories(self, **kwargs):
        return list(self.memories.values())

    def stats(self):
        return {"total_memories": len(self.memories)}

    def close(self):
        pass

    def execute(self, sql, params=None):
        return []

    def increment_access(self, ids):
        pass


class _FakeVectorStore:
    """Records every (memory_id, content) passed to add()."""

    ROLES = {"vector"}

    def __init__(self) -> None:
        self.adds: list[tuple[str, str, str]] = []

    def add(self, memory_id, content, namespace="default"):  # noqa: ANN001
        self.adds.append((memory_id, content, namespace))

    def search(self, query, limit=20, namespace=None):  # noqa: ANN001
        return []

    def count(self):
        return len(self.adds)

    def close(self):
        pass


def _build_mem_with_pii(*, mode: str = "off"):
    """Build an AgentMemory wired to fake stores + a PII config in the
    desired mode. Bypasses the registry/embedder probe."""
    import tempfile
    from unittest.mock import patch

    from attestor.compliance.pii import PIIConfig
    from attestor.core.agent_memory import AgentMemory

    fake_store = _FakeDocumentStore()
    fake_vec = _FakeVectorStore()

    tmp = tempfile.mkdtemp()
    with patch("attestor.core.agent_memory._registry") as _reg:
        _reg.return_value.DEFAULT_BACKENDS = ["postgres"]
        _reg.return_value.resolve_backends.return_value = {
            "document": "postgres",
            "vector": "vector",
        }

        def _instantiate(name, path, bcfg):
            if name == "postgres":
                return fake_store
            return fake_vec

        _reg.return_value.instantiate_backend.side_effect = _instantiate

        with patch(
            "attestor.store.embedder_dim_check.assert_embedder_dim_matches_schema",
            lambda *a, **k: None,
        ):
            mem = AgentMemory(tmp)

    # Force the PII config the test wants.
    mem._pii_cfg = PIIConfig(mode=mode, llm_model=None)

    return mem, fake_store, fake_vec


@pytest.mark.unit
def test_off_mode_unchanged_content() -> None:
    """``mode='off'`` → content untouched, no PII metadata, byte-identical
    to the legacy add() path."""
    mem, store, vec = _build_mem_with_pii(mode="off")
    try:
        m = mem.add(
            "Email jane@example.com about the deal.",
            namespace="default",
        )
    finally:
        mem.close()

    assert m.content == "Email jane@example.com about the deal."
    assert "_pii_findings" not in (m.metadata or {})
    assert "_pii_original_sealed" not in (m.metadata or {})
    # Vector store sees the original.
    assert vec.adds[-1][1] == "Email jane@example.com about the deal."


@pytest.mark.unit
def test_flag_mode_email() -> None:
    """``mode='flag'`` → 1 finding, content unchanged, metadata stamped."""
    mem, store, vec = _build_mem_with_pii(mode="flag")
    try:
        m = mem.add(
            "contact me at jane@example.com",
            namespace="default",
        )
    finally:
        mem.close()

    assert m.content == "contact me at jane@example.com"  # unchanged
    findings = (m.metadata or {}).get("_pii_findings")
    assert findings is not None
    assert isinstance(findings, list)
    assert len(findings) == 1
    assert findings[0]["type"] == "email"
    assert "_pii_original_sealed" not in (m.metadata or {})


@pytest.mark.unit
def test_flag_mode_multiple_pii_types() -> None:
    """Three different PII types → three findings on metadata."""
    mem, store, vec = _build_mem_with_pii(mode="flag")
    try:
        m = mem.add(
            "Email jane@example.com or (415) 555-1234. SSN 123-45-6789 secret.",
            namespace="default",
        )
    finally:
        mem.close()

    findings = (m.metadata or {}).get("_pii_findings")
    assert findings is not None
    types = {f["type"] for f in findings}
    assert "email" in types
    assert "phone" in types
    assert "ssn" in types


@pytest.mark.unit
def test_redact_mode_email() -> None:
    """``mode='redact'`` → content rewritten with [REDACTED:EMAIL]."""
    mem, store, vec = _build_mem_with_pii(mode="redact")
    try:
        m = mem.add(
            "Email: jane@example.com",
            namespace="default",
        )
    finally:
        mem.close()

    assert "[REDACTED:EMAIL]" in m.content
    assert "jane@example.com" not in m.content


@pytest.mark.unit
def test_redact_mode_preserves_structure_end_to_end() -> None:
    """Multiple PII items in same content all replaced; surrounding text intact."""
    mem, store, vec = _build_mem_with_pii(mode="redact")
    try:
        m = mem.add(
            "Reach Jane at jane@example.com or (415) 555-1234. SSN 123-45-6789.",
            namespace="default",
        )
    finally:
        mem.close()

    assert "jane@example.com" not in m.content
    assert "(415) 555-1234" not in m.content
    assert "123-45-6789" not in m.content
    assert "[REDACTED:EMAIL]" in m.content
    assert "[REDACTED:PHONE]" in m.content
    assert "[REDACTED:SSN]" in m.content
    assert "Reach Jane at" in m.content


@pytest.mark.unit
def test_redact_mode_seals_original() -> None:
    """``mode='redact'`` with findings → metadata['_pii_original_sealed']=True."""
    mem, store, vec = _build_mem_with_pii(mode="redact")
    try:
        m = mem.add(
            "Email: jane@example.com",
            namespace="default",
        )
    finally:
        mem.close()

    assert m.metadata.get("_pii_original_sealed") is True


@pytest.mark.unit
def test_recall_returns_redacted_content() -> None:
    """End-to-end: recall sees redacted content, not the original."""
    mem, store, vec = _build_mem_with_pii(mode="redact")
    try:
        mem.add("Email jane@example.com", namespace="default")
        # Pull straight from the document store via .get() — recall
        # needs a vector lane we haven't wired. Using the ID returned by add.
        ids = list(store.memories.keys())
        assert ids
        roundtrip = mem.get(ids[0])
    finally:
        mem.close()

    assert roundtrip is not None
    assert "jane@example.com" not in roundtrip.content
    assert "[REDACTED:EMAIL]" in roundtrip.content


@pytest.mark.unit
def test_round_trip_preserves_findings() -> None:
    """Findings metadata survives the document-store round trip."""
    mem, store, vec = _build_mem_with_pii(mode="flag")
    try:
        m = mem.add(
            "Email jane@example.com or call (415) 555-1234.",
            namespace="default",
        )
        roundtrip = mem.get(m.id)
    finally:
        mem.close()

    assert roundtrip is not None
    findings = (roundtrip.metadata or {}).get("_pii_findings")
    assert findings is not None
    types = {f["type"] for f in findings}
    assert "email" in types
    assert "phone" in types


@pytest.mark.unit
def test_detector_failure_isolated(monkeypatch, caplog) -> None:
    """If detect_pii raises, ingest still succeeds with _pii_detector_error
    metadata flag — failure isolation, never block the doc write path."""
    mem, store, vec = _build_mem_with_pii(mode="flag")

    def boom(*args, **kwargs):  # noqa: ANN001, ANN201
        raise RuntimeError("detector exploded")

    monkeypatch.setattr("attestor.compliance.pii.detect_pii", boom)
    monkeypatch.setattr("attestor.core.agent_memory.detect_pii", boom)

    try:
        with caplog.at_level(logging.WARNING):
            m = mem.add("Email jane@example.com", namespace="default")
    finally:
        mem.close()

    assert m is not None
    # Content untouched on failure (we didn't redact safely).
    assert m.content == "Email jane@example.com"
    # Failure flag is on the metadata so the operator can see it.
    assert m.metadata.get("_pii_detector_error") is True
    assert any("pii" in rec.message.lower() for rec in caplog.records)


@pytest.mark.unit
def test_llm_path_optional() -> None:
    """``mode='llm'`` calls a mocked LLM that surfaces names/addresses."""
    from attestor.compliance.pii import detect_pii

    fake_response = MagicMock()
    fake_response.choices = [
        MagicMock(message=MagicMock(content=(
            '[{"type":"name","span":[18,28],"confidence":0.85}]'
        ))),
    ]
    fake_response.id = "x"
    fake_response.model = "y"
    fake_response.usage = {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = fake_response

    text = "I had lunch with John Smith yesterday."
    result = detect_pii(
        text,
        mode="llm",
        llm_model="anthropic/claude-haiku-4.5",
        llm_client=fake_client,
    )
    types = {f.type for f in result.findings}
    assert "name" in types


# ─────────────────────────────────────────────────────────────────────────
# Smoke / regression corpus — false positive rate
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_pii_compliance_smoke_lme() -> None:
    """LME-style turn with email + phone is detected."""
    from attestor.compliance.pii import detect_pii

    turn = (
        "When I visited Tokyo last month, I emailed jane@example.com "
        "and texted (415) 555-1234 about the schedule."
    )
    result = detect_pii(turn, mode="flag")
    types = {f.type for f in result.findings}
    assert "email" in types
    assert "phone" in types


@pytest.mark.unit
def test_false_positive_rate_under_5pct_on_clean_corpus() -> None:
    """100+ LME-style natural-language samples without PII should produce
    a false-positive rate <5% — keeps the detector usable for opt-in
    enterprise deployments where every flagged sample is a queue cost."""
    from attestor.compliance.pii import detect_pii

    clean_corpus: list[str] = [
        # Travel / planning
        "I booked a flight to Tokyo last Friday.",
        "We're meeting at the office on Monday morning.",
        "The conference starts on October 14, 2024.",
        "Filed the report at 14:30 yesterday.",
        "Dinner reservation at 7:30 pm tomorrow.",
        "Pick up groceries on the way home.",
        "Walk the dog before the rain hits.",
        "The package arrived at 9:15 this morning.",
        "Set the alarm for 6:00 am.",
        "We left at sunset and got back at midnight.",
        # Coding / engineering
        "Refactored the authentication middleware today.",
        "Fixed the off-by-one error in the loop.",
        "Pushed the patch to the staging branch.",
        "The build completed in 2 minutes 30 seconds.",
        "Test coverage hit 87% on the new module.",
        "Deployed v2.3.1 of the service this morning.",
        "Migration ran in 14ms across 3 tables.",
        "Reduced p99 latency from 120ms to 45ms.",
        "Index size grew to 250MB after the bulk import.",
        "Indexed 1,200,000 documents overnight.",
        # Numbers + identifiers (NOT credit cards)
        "Order id 1234567890123456 logged successfully.",
        "Trace id abc123def456 corresponds to the failed request.",
        "Workflow run 9876543210987654 completed in 30s.",
        "The build hash is 8c7d6e5f4a3b2c1d9e0f.",
        "Container image digest sha256:abcdef1234567890.",
        # Dates
        "On 2024-01-15 we shipped the v3 release.",
        "Patient seen on 03/14/2024 at 10am.",
        "Valid through 12/31/2025 per the contract.",
        "Created at 2024-02-29 03:14:15 UTC.",
        "The 1990s were a transformative decade.",
        # Times
        "Standup at 9:00 sharp.",
        "Lunch from 12:00 to 13:00.",
        "Pager rotation 17:00-09:00 weekdays.",
        # Currency / amounts
        "Burn rate is around $42,000 per month.",
        "Revenue grew to $1.2M in Q3.",
        "Discount of 15% applied at checkout.",
        # Geography
        "Visited Shibuya crossing this morning.",
        "Stopped by Golden Gate Park on Saturday.",
        "Layover in Frankfurt was 90 minutes long.",
        "Drove from San Francisco to Los Angeles.",
        # Domain specific
        "The Postgres index runs cosine similarity on 1024-dim vectors.",
        "Pinecone Local serves localhost:5080 by default.",
        "Neo4j AuraDB hosts the graph backend.",
        "BM25 lane uses k1=1.2 and b=0.75 by default.",
        "MMR diversity uses lambda=0.7.",
        # Free-form
        "She mentioned her favorite color is blue.",
        "The team agreed on the architecture last sprint.",
        "Coffee tastes better at the small cafe down the street.",
        "Reading a book about distributed systems this weekend.",
        "The weather forecast shows rain through Thursday.",
        "His birthday is sometime in March.",
        "Annual offsite in the mountains every November.",
        "Replaced the keyboard last week — much smoother now.",
        "I prefer dark mode for code editors.",
        "The kanban board has 14 stories in progress.",
        # Phrasing that historically tripped phone/SSN regexes
        "Issue tracker URL ends with -123-45-6789 — confirmed not PII.",  # NOTE
        "Build number 30303030 was the long-running canary.",
        "Distance: 12,345 km.",
        "Temperature dropped to -3 degrees overnight.",
        # Misc
        "Pulled fresh bread from the oven.",
        "The puppy chewed through another shoe.",
        "Watching the new documentary tonight.",
        "Snowed all weekend in Tahoe.",
        "Traffic on the bridge is brutal today.",
        "Studied for the exam at the library until 11pm.",
        "Ran 5 miles before breakfast.",
        "The new restaurant opens next month.",
        "Sent the proposal for review by EOD Friday.",
        "Working remotely Wednesday and Thursday this week.",
        "Brought home flowers for the anniversary.",
        "Tried a new recipe — caramelized onions take forever.",
        "Movie night with the kids on Saturday.",
        "The garage door is sticking again.",
        "Ordered new running shoes online.",
        "Changed the air filter in the HVAC.",
        "Played chess for an hour after dinner.",
        "The book club meets every other Thursday.",
        "Dropped off donations at the local shelter.",
        "Replaced the smoke detector batteries.",
        "Fixed the leaky faucet in the kitchen.",
        "Repotted the basil this morning.",
        "Cleaned out the garage and sold three boxes.",
        "Adopted a kitten from the shelter last month.",
        "Tuned up the bike for the spring season.",
        "The HOA meeting ran until 8pm.",
        "Booked the rental cabin for July 4th weekend.",
        "Met with the financial planner yesterday.",
        "Listened to the podcast on the commute.",
        "Bought new running shoes — Brooks Ghost 15.",
        "Hiked the loop trail in 90 minutes.",
        "Joined a yoga class on Tuesday mornings.",
        "Rearranged the office furniture.",
        "Planted tomatoes in the side bed.",
        "Snowboarded all weekend at Tahoe.",
        "Visited the observatory after sunset.",
        "Read three chapters before bed.",
        "Tried the new pizza place downtown.",
        "Finished knitting the scarf this evening.",
        "Volunteered at the food bank Saturday.",
        "Cleaned out the pantry — donated everything expired.",
        "Reorganized the spice rack alphabetically.",
        "The kids built a fort in the living room.",
        "We binge-watched the new series last weekend.",
        "Went to the farmer's market early Sunday.",
        "Brought leftovers for lunch all week.",
        "Tinkered with the espresso machine settings.",
        "Tried sourdough — first loaf came out flat.",
        "Replaced the bathroom vanity light bulbs.",
        "Cleaned out the trunk — found two coats.",
    ]

    assert len(clean_corpus) >= 100, "corpus must be at least 100 samples"

    false_positives = 0
    for sample in clean_corpus:
        result = detect_pii(sample, mode="flag")
        if result.findings:
            false_positives += 1

    fp_rate = false_positives / len(clean_corpus)
    assert fp_rate < 0.05, (
        f"False positive rate {fp_rate:.2%} exceeds 5% on clean corpus "
        f"(found {false_positives} flags across {len(clean_corpus)} samples)"
    )
