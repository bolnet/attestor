"""v4 namespace round-trip — write w/ namespace, read it back.

Background — caught by the trace PR (#76):
    The v4 schema has no `namespace` column (replaced by user_id +
    project_id + scope for tenancy). Pre-fix, ``Memory.from_row``
    fell back to the dataclass default "default" for every v4 read,
    so any caller that used namespace as a metadata-style key (the
    LME bench writes namespace="lme_<sample>" for sample isolation)
    saw 100% of recall candidates dropped by the orchestrator's
    namespace filter.

    Fix:
      • core.add()  — stamp namespace into metadata["_namespace"]
                      when v4 is active and namespace != "default"
      • models.from_row — fall back to metadata["_namespace"] when
                      the row has no namespace column

These tests pin the round-trip without requiring a live Postgres —
they synthesize the row dicts that ``_row_to_memory`` consumes.
"""

from __future__ import annotations

import json

import pytest

from attestor.models import Memory


@pytest.mark.unit
def test_v4_row_with_metadata_namespace_round_trips():
    """A v4 row (no `namespace` column) carries the namespace inside
    metadata; from_row should recover it."""
    row = {
        "id": "00000000-0000-0000-0000-000000000001",
        "content": "v4 namespaced memory",
        "user_id": "00000000-0000-0000-0000-0000000000aa",
        "project_id": "00000000-0000-0000-0000-0000000000bb",
        "scope": "user",
        # NB: no `namespace` key — that's the v4 schema reality
        "metadata": {"_namespace": "lme_sample_42"},
        "tags": [],
        "category": "fact",
    }
    mem = Memory.from_row(row)
    assert mem.namespace == "lme_sample_42"


@pytest.mark.unit
def test_v4_row_with_metadata_as_json_string_round_trips():
    """Some Postgres drivers return jsonb columns as JSON strings (not
    dicts). from_row must parse the string before lookup."""
    row = {
        "id": "00000000-0000-0000-0000-000000000002",
        "content": "v4 string-metadata memory",
        "user_id": "00000000-0000-0000-0000-0000000000aa",
        "metadata": json.dumps({"_namespace": "tenant-acme"}),
        "tags": [],
    }
    mem = Memory.from_row(row)
    assert mem.namespace == "tenant-acme"


@pytest.mark.unit
def test_v4_row_without_namespace_metadata_falls_back_to_default():
    """v4 row + no _namespace in metadata = legacy "default". Pre-fix
    behavior preserved for the SOLO single-tenant happy path."""
    row = {
        "id": "00000000-0000-0000-0000-000000000003",
        "content": "v4 untenanted memory",
        "user_id": "00000000-0000-0000-0000-0000000000aa",
        "metadata": {},
        "tags": [],
    }
    mem = Memory.from_row(row)
    assert mem.namespace == "default"


@pytest.mark.unit
def test_v3_row_top_level_namespace_still_wins():
    """v3 schema kept the top-level namespace column. We must not
    regress v3 reads — the column trumps any metadata fallback."""
    row = {
        "id": "v3id000000aa",
        "content": "v3 namespaced memory",
        "namespace": "v3-tenant",
        "metadata": {"_namespace": "would-have-been-this-but-v3-wins"},
        "tags": [],
    }
    mem = Memory.from_row(row)
    assert mem.namespace == "v3-tenant"


@pytest.mark.unit
def test_v4_row_with_non_dict_metadata_falls_back_safely():
    """If metadata happens to be a JSON string that parses to a list
    (legacy junk data), we must not crash on .get()."""
    row = {
        "id": "00000000-0000-0000-0000-000000000005",
        "content": "v4 list-metadata memory",
        "user_id": "00000000-0000-0000-0000-0000000000aa",
        "metadata": json.dumps(["why", "is", "this", "a", "list"]),
        "tags": [],
    }
    mem = Memory.from_row(row)
    assert mem.namespace == "default"


@pytest.mark.unit
def test_v4_row_with_explicit_default_namespace_in_metadata():
    """Caller stamped metadata["_namespace"]="default" — round-trips
    as "default" (no surprise). Same as the no-metadata case."""
    row = {
        "id": "00000000-0000-0000-0000-000000000006",
        "content": "v4 explicit-default memory",
        "user_id": "00000000-0000-0000-0000-0000000000aa",
        "metadata": {"_namespace": "default"},
        "tags": [],
    }
    mem = Memory.from_row(row)
    assert mem.namespace == "default"
