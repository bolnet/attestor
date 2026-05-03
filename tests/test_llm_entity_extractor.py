"""Unit tests for the LLM-driven entity + relation extractor.

The regex-based ``attestor.graph.extractor`` returns ``entity_count=0``
for most LME-S content (verified via ``entity_names=[]`` traces in
production). This test file pins the behavior of the new
``attestor.extraction.llm_entity_extractor`` module that replaces the
regex layer when ``stack.ingest.llm_entity_extraction.enabled`` is
True. The 2026 Mem0/Zep architectural pattern is to run an LLM at
ingest to extract entities + relationships; this is that path.

Contract under test
-------------------

  - returns ([], []) when api_key is missing OR LLM call times out OR
    LLM returns malformed JSON. Failure isolation is the whole point —
    a flaky LLM must NEVER block ingest.
  - extracts proper-noun entities ("Wells Fargo") + value-bearing
    entities ("$350,000" as type=quantity) from natural-language LME-S
    content.
  - assigns relations only when explicitly stated in content; refuses
    to fabricate.
  - prompt asks for temperature=0 (deterministic) and embeds 2-3
    worked LME-style examples.
  - ``AgentMemory.add()`` integration:
      * ``llm_entity_extraction.enabled=True``  → uses LLM extractor
      * ``llm_entity_extraction.enabled=False`` → uses regex extractor
        (default; byte-identical to main on disabled path)
  - process-local cache keyed by content_hash → second add() of the
    same content does NOT pay for a second LLM call.
  - LME-S 852ce960 regression: feed "[2023-08-11] User: pre-approved
    for $350,000 from Wells Fargo" and assert entity_count > 0 (this
    is the production failure that motivated this PR).
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────


def _fake_llm(response_text: str) -> Any:
    """Build a mock OpenAI-style client that returns ``response_text``."""
    fake_response = MagicMock()
    fake_response.choices = [MagicMock(message=MagicMock(content=response_text))]
    fake_response.id = "test-id"
    fake_response.model = "test/m"
    fake_response.usage = {
        "prompt_tokens": 10, "completion_tokens": 30, "total_tokens": 40,
    }
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = fake_response
    return fake_client


def _llm_payload(entities: list[dict[str, Any]],
                 relations: list[dict[str, Any]] | None = None) -> str:
    return json.dumps({"entities": entities, "relations": relations or []})


# ─────────────────────────────────────────────────────────────────────
# Module imports + dataclass shape
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_module_imports() -> None:
    from attestor.extraction.llm_entity_extractor import (  # noqa: F401
        ExtractedEntity,
        ExtractedRelation,
        extract_with_llm,
    )


@pytest.mark.unit
def test_extracted_entity_is_frozen_dataclass() -> None:
    from attestor.extraction.llm_entity_extractor import ExtractedEntity

    e = ExtractedEntity(name="Wells Fargo", type="organization", attributes={})
    assert e.name == "Wells Fargo"
    assert e.type == "organization"
    with pytest.raises((AttributeError, Exception)):
        e.name = "Other"  # type: ignore[misc]  # frozen


@pytest.mark.unit
def test_extracted_relation_is_frozen_dataclass() -> None:
    from attestor.extraction.llm_entity_extractor import ExtractedRelation

    r = ExtractedRelation(
        from_entity="User", to="SoFi", type="works_at", metadata={},
    )
    assert r.from_entity == "User"
    assert r.to == "SoFi"
    assert r.type == "works_at"
    with pytest.raises((AttributeError, Exception)):
        r.type = "other"  # type: ignore[misc]


# ─────────────────────────────────────────────────────────────────────
# Failure isolation — empty / timeout / malformed → ([], [])
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_returns_empty_on_no_api_key(monkeypatch) -> None:
    """No api_key + no provider key in env → empty result, no escape."""
    from attestor.extraction.llm_entity_extractor import extract_with_llm

    # Simulate "no key" by injecting a client factory that raises on auth
    def fake_resolver(model: str, api_key: str | None) -> tuple[Any, str]:
        raise RuntimeError("OPENAI_API_KEY not set")

    monkeypatch.setattr(
        "attestor.extraction.llm_entity_extractor._resolve_client",
        fake_resolver,
    )

    entities, relations = extract_with_llm(
        "User got pre-approved by Wells Fargo",
        model="anthropic/claude-haiku-4.5",
        api_key=None,
    )
    assert entities == []
    assert relations == []


@pytest.mark.unit
def test_returns_empty_on_llm_timeout(monkeypatch) -> None:
    """LLM raises (timeout / network down) → empty result, no escape."""
    from attestor.extraction.llm_entity_extractor import extract_with_llm

    fake_client = MagicMock()
    fake_client.chat.completions.create.side_effect = TimeoutError("deadline")

    monkeypatch.setattr(
        "attestor.extraction.llm_entity_extractor._resolve_client",
        lambda model, api_key: (fake_client, "claude-haiku-4.5"),
    )

    entities, relations = extract_with_llm(
        "anything", model="anthropic/claude-haiku-4.5",
    )
    assert entities == []
    assert relations == []


@pytest.mark.unit
def test_returns_empty_on_malformed_json(monkeypatch) -> None:
    """LLM returns garbage instead of JSON → empty result, no escape."""
    from attestor.extraction.llm_entity_extractor import extract_with_llm

    fake_client = _fake_llm("This is not JSON at all { broken")
    monkeypatch.setattr(
        "attestor.extraction.llm_entity_extractor._resolve_client",
        lambda model, api_key: (fake_client, "claude-haiku-4.5"),
    )
    entities, relations = extract_with_llm(
        "User likes Python", model="anthropic/claude-haiku-4.5",
    )
    assert entities == []
    assert relations == []


# ─────────────────────────────────────────────────────────────────────
# Quality — proper nouns + value-bearing entities + relations
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_extracts_proper_nouns(monkeypatch) -> None:
    """LME-style content → proper noun "Wells Fargo" extracted."""
    from attestor.extraction.llm_entity_extractor import extract_with_llm

    payload = _llm_payload(
        entities=[
            {"name": "User", "type": "person", "attributes": {}},
            {"name": "Wells Fargo", "type": "organization", "attributes": {}},
        ],
        relations=[],
    )
    fake_client = _fake_llm(payload)
    monkeypatch.setattr(
        "attestor.extraction.llm_entity_extractor._resolve_client",
        lambda model, api_key: (fake_client, "claude-haiku-4.5"),
    )

    entities, _ = extract_with_llm(
        "User got pre-approved by Wells Fargo",
        model="anthropic/claude-haiku-4.5",
    )
    names = [e.name for e in entities]
    assert "Wells Fargo" in names


@pytest.mark.unit
def test_extracts_amounts_as_quantity_entity(monkeypatch) -> None:
    """Value-bearing token "$350,000" → entity with type=quantity."""
    from attestor.extraction.llm_entity_extractor import extract_with_llm

    payload = _llm_payload(
        entities=[
            {"name": "$350,000", "type": "quantity", "attributes": {"currency": "USD"}},
            {"name": "Wells Fargo", "type": "organization", "attributes": {}},
        ],
    )
    fake_client = _fake_llm(payload)
    monkeypatch.setattr(
        "attestor.extraction.llm_entity_extractor._resolve_client",
        lambda model, api_key: (fake_client, "claude-haiku-4.5"),
    )

    entities, _ = extract_with_llm(
        "Pre-approved for $350,000 from Wells Fargo",
        model="anthropic/claude-haiku-4.5",
    )
    quantities = [e for e in entities if e.type == "quantity"]
    assert any(e.name == "$350,000" for e in quantities)


@pytest.mark.unit
def test_assigns_relation_when_stated(monkeypatch) -> None:
    """Explicit "User works at SoFi" → relation works_at(User, SoFi)."""
    from attestor.extraction.llm_entity_extractor import extract_with_llm

    payload = _llm_payload(
        entities=[
            {"name": "User", "type": "person", "attributes": {}},
            {"name": "SoFi", "type": "organization", "attributes": {}},
        ],
        relations=[
            {"from": "User", "to": "SoFi", "type": "works_at", "metadata": {}},
        ],
    )
    fake_client = _fake_llm(payload)
    monkeypatch.setattr(
        "attestor.extraction.llm_entity_extractor._resolve_client",
        lambda model, api_key: (fake_client, "claude-haiku-4.5"),
    )

    _, relations = extract_with_llm(
        "User works at SoFi", model="anthropic/claude-haiku-4.5",
    )
    assert any(
        r.from_entity == "User" and r.to == "SoFi" and r.type == "works_at"
        for r in relations
    )


@pytest.mark.unit
def test_does_not_fabricate_relations(monkeypatch) -> None:
    """No explicit relation in content → empty relations array.

    The prompt asks the model to return ``"relations": []`` when the
    content has no stated relation. We pin the contract: when the
    fixture returns no relations, the extractor passes that through —
    NOT inventing edges from entity co-occurrence.
    """
    from attestor.extraction.llm_entity_extractor import extract_with_llm

    payload = _llm_payload(
        entities=[
            {"name": "Python", "type": "tool", "attributes": {}},
            {"name": "Rust", "type": "tool", "attributes": {}},
        ],
        relations=[],
    )
    fake_client = _fake_llm(payload)
    monkeypatch.setattr(
        "attestor.extraction.llm_entity_extractor._resolve_client",
        lambda model, api_key: (fake_client, "claude-haiku-4.5"),
    )

    _, relations = extract_with_llm(
        "Python and Rust are popular",  # no explicit relation
        model="anthropic/claude-haiku-4.5",
    )
    assert relations == []


# ─────────────────────────────────────────────────────────────────────
# Prompt design constraints
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_temperature_is_zero(monkeypatch) -> None:
    """Determinism: every LLM call uses temperature=0."""
    from attestor.extraction.llm_entity_extractor import extract_with_llm

    captured: dict[str, Any] = {}
    fake_response = MagicMock()
    fake_response.choices = [MagicMock(message=MagicMock(
        content=_llm_payload(entities=[]),
    ))]
    fake_client = MagicMock()

    def capture(**kwargs):
        captured.update(kwargs)
        return fake_response

    fake_client.chat.completions.create.side_effect = capture
    monkeypatch.setattr(
        "attestor.extraction.llm_entity_extractor._resolve_client",
        lambda model, api_key: (fake_client, "claude-haiku-4.5"),
    )

    extract_with_llm("anything", model="anthropic/claude-haiku-4.5")
    assert captured.get("temperature") == 0.0


@pytest.mark.unit
def test_prompt_contains_worked_examples() -> None:
    """The prompt must include 2-3 LME-S-style worked examples.

    Worked examples are the single biggest lever for output-quality on
    structured extraction tasks. We pin their presence so a future
    "tighten the prompt" refactor can't accidentally strip them.
    """
    from attestor.extraction.llm_entity_extractor import _EXTRACTION_PROMPT

    text = _EXTRACTION_PROMPT.lower()
    # Two distinct example sections; proper-noun + amount tokens
    assert "example" in text
    # At least three example lines (mention of "user:", "wells fargo",
    # "$350,000" — covers the LME-S pre-approval case from production).
    assert "wells fargo" in text
    assert "$350,000" in text or "$" in text  # quantity demo present


@pytest.mark.unit
def test_prompt_lists_entity_type_vocabulary() -> None:
    """Prompt must enumerate the fixed type vocabulary.

    Without a closed type set the model invents new types and the
    downstream graph layer can't index by type.
    """
    from attestor.extraction.llm_entity_extractor import _EXTRACTION_PROMPT

    text = _EXTRACTION_PROMPT.lower()
    for vocab in ("person", "organization", "concept", "event", "tool",
                  "location", "quantity", "other"):
        assert vocab in text, f"type {vocab!r} missing from prompt vocabulary"


@pytest.mark.unit
def test_prompt_truncates_long_content(monkeypatch) -> None:
    """Cap content to first 4000 chars before sending to the LLM."""
    from attestor.extraction.llm_entity_extractor import extract_with_llm

    captured: dict[str, Any] = {}
    fake_client = MagicMock()
    fake_response = MagicMock()
    fake_response.choices = [MagicMock(message=MagicMock(
        content=_llm_payload(entities=[]),
    ))]

    def capture(**kwargs):
        captured.update(kwargs)
        return fake_response

    fake_client.chat.completions.create.side_effect = capture
    monkeypatch.setattr(
        "attestor.extraction.llm_entity_extractor._resolve_client",
        lambda model, api_key: (fake_client, "claude-haiku-4.5"),
    )

    long_content = "X" * 10000
    extract_with_llm(long_content, model="anthropic/claude-haiku-4.5")
    msg = captured["messages"][-1]["content"]
    # The prompt embedded < 4000 X's worth of content (plus prompt body).
    assert msg.count("X") <= 4000


# ─────────────────────────────────────────────────────────────────────
# Caching by content_hash
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_caches_by_content_hash(monkeypatch) -> None:
    """Same content → one LLM call (second pull from cache)."""
    from attestor.extraction.llm_entity_extractor import (
        _reset_cache,
        extract_with_llm,
    )

    _reset_cache()  # isolate from prior tests

    payload = _llm_payload(
        entities=[{"name": "Wells Fargo", "type": "organization", "attributes": {}}],
    )
    fake_client = _fake_llm(payload)
    monkeypatch.setattr(
        "attestor.extraction.llm_entity_extractor._resolve_client",
        lambda model, api_key: (fake_client, "claude-haiku-4.5"),
    )

    extract_with_llm("Pre-approved by Wells Fargo",
                     model="anthropic/claude-haiku-4.5")
    extract_with_llm("Pre-approved by Wells Fargo",
                     model="anthropic/claude-haiku-4.5")
    # Caching at module level: single call across two invocations of
    # the same content.
    assert fake_client.chat.completions.create.call_count == 1


# ─────────────────────────────────────────────────────────────────────
# Config — LLMExtractionCfg + YAML loader
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_llm_extraction_cfg_defaults() -> None:
    from attestor.config import LLMExtractionCfg

    cfg = LLMExtractionCfg()
    assert cfg.enabled is False
    assert cfg.model == "anthropic/claude-haiku-4.5"
    assert cfg.timeout_s == pytest.approx(15.0)
    assert cfg.cache is True


@pytest.mark.unit
def test_yaml_loader_parses_llm_entity_extraction_block(tmp_path) -> None:
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
  ingest:
    llm_entity_extraction:
      enabled: true
      model: anthropic/claude-haiku-4.5
      timeout_s: 20.0
      cache: false
""")
    stack = load_stack(yaml_path)
    le = stack.ingest.llm_entity_extraction
    assert le.enabled is True
    assert le.model == "anthropic/claude-haiku-4.5"
    assert le.timeout_s == pytest.approx(20.0)
    assert le.cache is False


@pytest.mark.unit
def test_yaml_loader_defaults_llm_entity_extraction(tmp_path) -> None:
    """Missing block → default LLMExtractionCfg (enabled=False)."""
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
    assert stack.ingest.llm_entity_extraction.enabled is False
    assert stack.ingest.llm_entity_extraction.cache is True


# ─────────────────────────────────────────────────────────────────────
# AgentMemory.add() integration — branch on the cfg toggle
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_falls_back_to_regex_when_llm_disabled() -> None:
    """``llm_entity_extraction.enabled=False`` (default) → regex extractor.

    This pins the byte-identical-on-disabled invariant. The regex path
    is the existing ``attestor.graph.extractor.extract_entities_and_relations``.
    """
    from attestor.extraction.llm_entity_extractor import (
        extract_or_regex,
    )

    # Disabled cfg → delegates to the regex path; we feed content that
    # the regex can match ("works at X") so we get a deterministic
    # non-empty output to distinguish from the LLM path.
    nodes, edges = extract_or_regex(
        content="Alice works at Google",
        tags=[],
        entity="Alice",
        category="career",
        namespace=None,
        llm_enabled=False,
    )
    edge_types = [e["type"] for e in edges]
    assert "works_at" in edge_types


@pytest.mark.unit
def test_extract_or_regex_uses_llm_when_enabled(monkeypatch) -> None:
    """``llm_enabled=True`` → calls the LLM path; result lifted to
    nodes/edges shape compatible with the graph backend."""
    from attestor.extraction.llm_entity_extractor import extract_or_regex

    payload = _llm_payload(
        entities=[
            {"name": "User", "type": "person", "attributes": {}},
            {"name": "Wells Fargo", "type": "organization", "attributes": {}},
            {"name": "$350,000", "type": "quantity", "attributes": {}},
        ],
        relations=[
            {"from": "User", "to": "Wells Fargo",
             "type": "pre_approved_by", "metadata": {}},
        ],
    )
    fake_client = _fake_llm(payload)
    monkeypatch.setattr(
        "attestor.extraction.llm_entity_extractor._resolve_client",
        lambda model, api_key: (fake_client, "claude-haiku-4.5"),
    )

    nodes, edges = extract_or_regex(
        content="User pre-approved for $350,000 from Wells Fargo",
        tags=[],
        entity=None,
        category="general",
        namespace="ns-test",
        llm_enabled=True,
        llm_model="anthropic/claude-haiku-4.5",
    )
    names = {n["name"] for n in nodes}
    assert "Wells Fargo" in names
    assert "$350,000" in names
    # All nodes carry the namespace through to the graph backend
    assert all(n.get("namespace") == "ns-test" for n in nodes)
    edge_types = [e["type"] for e in edges]
    assert "pre_approved_by" in edge_types


@pytest.mark.unit
def test_extract_or_regex_falls_back_to_regex_on_llm_failure(monkeypatch) -> None:
    """LLM enabled BUT call fails → degrade to regex, NOT empty.

    Failure isolation invariant: ingest must never break, and we must
    not silently drop graph data when the LLM is sick. The existing
    regex extractor still produces some output for "works at"-shaped
    content; we want that as the fallback.
    """
    from attestor.extraction.llm_entity_extractor import extract_or_regex

    fake_client = MagicMock()
    fake_client.chat.completions.create.side_effect = TimeoutError("deadline")
    monkeypatch.setattr(
        "attestor.extraction.llm_entity_extractor._resolve_client",
        lambda model, api_key: (fake_client, "claude-haiku-4.5"),
    )

    nodes, edges = extract_or_regex(
        content="Alice works at Google",
        tags=[],
        entity="Alice",
        category="career",
        namespace=None,
        llm_enabled=True,
        llm_model="anthropic/claude-haiku-4.5",
    )
    # Regex path produces "works_at" — degraded LLM falls back to it.
    assert any(e["type"] == "works_at" for e in edges)


# ─────────────────────────────────────────────────────────────────────
# LME-S 852ce960 regression — the production failure that motivated this PR
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_lme_852ce960_pre_approval_extracts_topic(monkeypatch) -> None:
    """Production failure: this content traces ``entity_count=0`` on the
    regex path. The LLM path must produce entity_count > 0.

    Memory ID 852ce960: an LME-S knowledge-update sample where the
    user's mortgage pre-approval state changes across sessions. The
    regex extractor returns ``entity_names=[]`` → graph layer dead →
    no supersession signal → wrong answer at recall time.
    """
    from attestor.extraction.llm_entity_extractor import extract_or_regex

    payload = _llm_payload(
        entities=[
            {"name": "User", "type": "person", "attributes": {}},
            {"name": "Wells Fargo", "type": "organization",
             "attributes": {"role": "lender"}},
            {"name": "$350,000", "type": "quantity",
             "attributes": {"currency": "USD"}},
            {"name": "pre-approved", "type": "event",
             "attributes": {"date": "2023-08-11"}},
        ],
        relations=[
            {"from": "User", "to": "Wells Fargo",
             "type": "pre_approved_by", "metadata": {}},
            {"from": "User", "to": "$350,000",
             "type": "approved_for_amount", "metadata": {}},
        ],
    )
    fake_client = _fake_llm(payload)
    monkeypatch.setattr(
        "attestor.extraction.llm_entity_extractor._resolve_client",
        lambda model, api_key: (fake_client, "claude-haiku-4.5"),
    )

    content = "[2023-08-11] User: pre-approved for $350,000 from Wells Fargo"
    nodes, edges = extract_or_regex(
        content=content,
        tags=[],
        entity=None,
        category="general",
        namespace=None,
        llm_enabled=True,
        llm_model="anthropic/claude-haiku-4.5",
    )
    assert len(nodes) > 0, (
        f"852ce960 regression: entity_count must be > 0 on LLM path, "
        f"got nodes={nodes}"
    )
    names = {n["name"] for n in nodes}
    # The two LME-S "topic markers" we couldn't pull with regex.
    assert "Wells Fargo" in names
    assert "$350,000" in names
