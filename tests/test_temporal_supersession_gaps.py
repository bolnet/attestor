"""Regression tests for the supersession-layer bugs surfaced by LME-S
knowledge-update sample 852ce960 (Wells Fargo $400k mortgage).

Each test in this file maps 1:1 to a production bug uncovered during the
2026-05-02 deep-dive. The bugs were invisible because the existing test
suite either:

  - never asserted ``valid_from`` reflects ``event_date``
  - asserted the entity-None short-circuit as DESIRED behavior
  - used only string-different content (paraphrase conflicts untested)
  - wrote v4 round-trip tests without ``event_date`` in the payload
  - had no multi-session "same date, different value" scenario

Tests that would currently REGRESS production are marked ``xfail`` with
``strict=True`` so flipping the underlying behavior surfaces immediately
when the corresponding fix lands.

See ``project_lme_ku_supersession_bugs_20260502.md`` in long-term memory
for the full bug taxonomy.
"""

from __future__ import annotations

import uuid

import pytest

# `mem` fixture comes from conftest.py — requires live Postgres (POSTGRES_URL).
# Each test uses a per-test unique ``entity`` token so the shared test DB
# doesn't bleed state across tests (the fixture intentionally does not
# truncate between runs to keep the live-stack assumption simple).


def _tag() -> str:
    """Per-test unique entity/category suffix to isolate state."""
    return uuid.uuid4().hex[:10]


# ── Gap 1 — valid_from must track event_date ──────────────────────────────


@pytest.mark.integration
def test_add_with_event_date_sets_valid_from(mem) -> None:
    """``mem.add(event_date=X)`` must produce ``memory.valid_from == X``.

    Production bug: the LME ingest path passes ``event_date=iso`` but
    ``valid_from`` always defaults to ``NOW()``. Result: every memory looks
    like it happened at ingest time, the temporal manager can't order
    same-day events, and supersession breaks for knowledge-update.
    """
    ent = f"mortgage_{_tag()}"
    m = mem.add(
        "pre-approved for $350,000 from Wells Fargo",
        category="finance",
        entity=ent,
        event_date="2023-08-11T00:00:00+00:00",
    )
    fetched = mem.get(m.id)
    assert fetched is not None
    assert fetched.valid_from.startswith("2023-08-11"), (
        f"valid_from {fetched.valid_from!r} did not track event_date "
        f"{fetched.event_date!r}"
    )


# ── Gap 2 — entity-None fallback via content-skeleton matching ───────────


@pytest.mark.integration
def test_supersession_falls_back_when_entity_missing(mem) -> None:
    """When the extractor fails to tag an entity, the temporal manager falls
    back to grouping by content skeleton (numeric-stripped). Two memories
    with the same template but different values supersede correctly.

    Anti-regression: distinct-skeleton entity-None pairs (e.g. "Likes Python"
    vs "Likes JavaScript") must NOT trigger this — covered by
    ``test_no_entity_no_implicit_supersession_but_contradiction_visible``
    below and by ``test_no_contradiction_without_entity`` in test_temporal.py.
    """
    cat = f"finance_{_tag()}"
    m1 = mem.add("pre-approved for $350,000", category=cat)
    mem.add("pre-approved for $400,000", category=cat)
    assert mem.get(m1.id).status == "superseded"


# ── Gap 3 — entity-tagged same-date supersession (regression guard) ─────


@pytest.mark.integration
def test_same_date_entity_tagged_supersedes_correctly(mem) -> None:
    """Pin the working entity-tagged supersession path so a future change
    that breaks it surfaces here. This currently works because the string
    comparator on different amounts ('$350,000' vs '$400,000') triggers
    supersession even though the date anchor is the same.

    The real production failure (Gap 5 below) is when the LME extractor
    fails to tag an entity — same content shape, no entity, supersession
    short-circuits.
    """
    ent = f"mortgage_{_tag()}"
    cat = f"finance_{_tag()}"
    mem.add(
        "pre-approved for $350,000",
        category=cat, entity=ent,
        event_date="2023-08-11T00:00:00+00:00",
    )
    newer = mem.add(
        "pre-approved for $400,000",
        category=cat, entity=ent,
        event_date="2023-08-11T00:00:00+00:00",
    )
    current = [
        m for m in mem.search(category=cat, entity=ent)
        if m.status == "active"
    ]
    assert len(current) == 1
    assert current[0].id == newer.id


# ── Gap 4 — v4 INSERT silently drops event_date ──────────────────────────


@pytest.mark.integration
def test_event_date_round_trips_through_store(mem) -> None:
    """``Memory.event_date`` must survive a write+read round-trip.

    Pre-fix: in v4 mode the INSERT statement at
    ``attestor/store/_postgres_document.py`` had no ``event_date`` column so
    the value was silently discarded. Post-fix: when ``event_date`` is
    supplied, the same value lands in ``valid_from`` so downstream readers
    have a real anchor regardless of the underlying schema.
    """
    ent = f"mortgage_{_tag()}"
    m = mem.add(
        "Wells Fargo pre-approval",
        category="finance", entity=ent,
        event_date="2023-08-11T00:00:00+00:00",
    )
    fetched = mem.get(m.id)
    assert fetched is not None
    assert fetched.valid_from.startswith("2023-08-11"), (
        "post-fix invariant: caller-supplied event_date must be preserved on "
        f"valid_from regardless of v3/v4 schema; got {fetched.valid_from!r}"
    )


# ── Gap 5 — multi-session same-date, entity-NULL, cross-template ─────────


@pytest.mark.integration
def test_multi_session_same_date_supersession_no_entity(mem) -> None:
    """The literal LME 852ce960 production scenario — no entity, same date,
    cross-template paraphrase. Closes via the auto-topic top-K
    intersection: m1's topic set {"wells", "fargo", "preapprov"} and m2's
    {"bump", "preapprov"} share "preapprov", which anchors them as the
    same fact for supersession.

    Anti-regression:
    - vector-similarity won't work here (measured 2026-05-03: structurally
      similar preferences are CLOSER than semantic paraphrases on
      llama-text-embed-v2; see project_lme_ku_semantic_threshold_null
      memory). Don't replace top-K intersection with cosine search.
    - "Likes Python" vs "Likes JavaScript" must NOT trigger this — covered
      by ``test_no_contradiction_without_entity`` and Gap 6 below.
    """
    cat = f"conversation_{_tag()}"
    mem.add(
        "[2023-08-11] User: I'm pre-approved for $350,000 from Wells Fargo",
        category=cat,
        event_date="2023-08-11T00:00:00+00:00",
    )
    mem.add(
        "[2023-08-11] User: My pre-approval was bumped to $400,000",
        category=cat,
        event_date="2023-08-11T00:00:00+00:00",
    )
    actives = [m for m in mem.search(category=cat) if m.status == "active"]
    assert len(actives) == 1, (
        f"expected one current pre-approval; got {len(actives)} "
        f"actives — the supersession layer didn't fire"
    )
    assert "$400" in actives[0].content


# ── Gap 6 — replace the bug-asserting test with the correct one ──────────


@pytest.mark.integration
def test_no_entity_no_implicit_supersession_but_contradiction_visible(mem) -> None:
    """Without an entity, supersession can't fire automatically — that's a
    real product limitation, not a bug. But the documented contract should
    be that the caller can detect the missing supersession by inspecting
    ``mem.search()`` and seeing both rows still active. This test pins the
    documented contract so a future change has to update the contract too,
    instead of the silent-drift-with-no-signal behaviour we have now.
    """
    cat = f"preference_{_tag()}"
    m1 = mem.add("Likes Python", category=cat)
    m2 = mem.add("Likes JavaScript", category=cat)
    assert mem.get(m1.id).status == "active"
    assert mem.get(m2.id).status == "active"
    actives = [m for m in mem.search(category=cat) if m.status == "active"]
    assert len(actives) >= 2, (
        "with entity=None, supersession is skipped — caller must be able "
        "to see both memories still active to detect the gap"
    )


# ── Gap 7 — timeline must use real event time, not insertion order ───────


@pytest.mark.integration
def test_timeline_orders_by_event_time_not_insertion_order(mem) -> None:
    """``timeline()`` sorts by event time. Pre-fix: ``valid_from`` always =
    ingest time, so even with the ``event_date or created_at`` fallback in
    ``manager.py:27`` the order was insertion-order. Post-fix: ``valid_from``
    carries the real event time and timeline reflects it.

    Insert in REVERSE chronological order — if the system orders by ingest
    time the result is wrong; ordering by real event time gives the right
    answer.
    """
    ent = f"life_{_tag()}"
    cat = f"event_{_tag()}"
    mem.add(
        "second event",
        entity=ent, category=cat,
        event_date="2023-06-01T00:00:00+00:00",
    )
    mem.add(
        "first event",
        entity=ent, category=cat,
        event_date="2023-01-01T00:00:00+00:00",
    )
    timeline = mem.timeline(ent)
    assert len(timeline) == 2
    contents = [m.content for m in timeline]
    assert contents == ["first event", "second event"], (
        f"expected chronological order by event time, got {contents!r}"
    )
