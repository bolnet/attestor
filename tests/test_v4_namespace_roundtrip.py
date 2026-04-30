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


# ──────────────────────────────────────────────────────────────────────
# Search-side filter: pin that v4 search filters on metadata->>'_namespace'
#
# Background — the missing companion to the read-side fix:
#     PR #77 made namespace round-trip through metadata["_namespace"] on
#     the WRITE path. But ``PostgresBackend.search()`` and
#     ``list_memories()`` skipped the namespace filter entirely on v4
#     ("replaced by RLS" — the comment was misleading; RLS at memories
#     filters by user_id, NOT namespace). LME-S benchmarks (one bench
#     user + many per-sample namespaces) returned cross-sample
#     contaminated candidates, blowing up retrieval verdicts.
# ──────────────────────────────────────────────────────────────────────


def _build_v4_backend_stub(captured_sql: list, captured_params: list):
    """Construct a PostgresBackend in v4 mode without going through
    schema init. _execute is stubbed to capture (sql, params) so the
    test can assert on the WHERE clause.
    """
    from attestor.store.postgres_backend import PostgresBackend

    backend = PostgresBackend.__new__(PostgresBackend)
    backend._v4 = True

    def _capture(sql, params=None):
        captured_sql.append(sql)
        captured_params.append(params)
        return []

    backend._execute = _capture
    backend._embed = lambda text: [0.0] * 1024  # deterministic stub
    return backend


def _build_v3_backend_stub(captured_sql: list, captured_params: list):
    from attestor.store.postgres_backend import PostgresBackend

    backend = PostgresBackend.__new__(PostgresBackend)
    backend._v4 = False
    backend._execute = lambda sql, params=None: (
        captured_sql.append(sql) or captured_params.append(params) or []
    )
    backend._embed = lambda text: [0.0] * 1024
    return backend


@pytest.mark.unit
def test_v4_search_filters_on_metadata_namespace():
    """Vector search MUST add ``metadata->>'_namespace' = %s`` when v4
    + namespace passed. Pre-fix, this filter was skipped → bench saw
    cross-namespace contamination across 80+ samples."""
    sqls: list = []
    params_log: list = []
    backend = _build_v4_backend_stub(sqls, params_log)

    backend.search("any query", namespace="lme_sample_42")

    assert sqls, "search did not call _execute"
    sql = sqls[0]
    assert "metadata->>'_namespace' = %s" in sql, (
        f"v4 search must filter on metadata->>'_namespace', got SQL: {sql}"
    )
    # The namespace value is the second positional param (after the
    # query vector embedded as %s::vector).
    assert "lme_sample_42" in params_log[0], (
        f"namespace must be passed as a param, got: {params_log[0]}"
    )


@pytest.mark.unit
def test_v3_search_filters_on_namespace_column():
    """v3 keeps the original ``namespace = %s`` filter (top-level
    column). Regression guard for legacy v3 deployments."""
    sqls: list = []
    params_log: list = []
    backend = _build_v3_backend_stub(sqls, params_log)

    backend.search("any query", namespace="legacy-tenant")

    sql = sqls[0]
    assert "namespace = %s" in sql, (
        f"v3 search must filter on top-level namespace, got SQL: {sql}"
    )
    assert "metadata->>" not in sql, (
        f"v3 search must NOT use metadata jsonb path, got SQL: {sql}"
    )


@pytest.mark.unit
def test_v4_search_no_namespace_filter_when_namespace_omitted():
    """Calling search() without a namespace should produce NO namespace
    filter on either schema — preserves the SOLO single-tenant happy
    path where every namespace is "default" and an explicit filter
    would be needless overhead."""
    sqls: list = []
    params_log: list = []
    backend = _build_v4_backend_stub(sqls, params_log)

    backend.search("any query", namespace=None)

    sql = sqls[0]
    assert "metadata->>'_namespace'" not in sql
    assert "namespace = " not in sql


@pytest.mark.unit
def test_v4_list_memories_filters_on_metadata_namespace():
    """Same fix applied to the list_memories() (temporal/category
    filter path). Pre-fix, list_memories with namespace silently
    returned ALL of the user's memories on v4."""
    sqls: list = []
    params_log: list = []
    backend = _build_v4_backend_stub(sqls, params_log)

    backend.list_memories(namespace="tenant-acme")

    assert sqls
    sql = sqls[0]
    assert "metadata->>'_namespace' = %(namespace)s" in sql, (
        f"v4 list_memories must filter on metadata->>'_namespace', "
        f"got SQL: {sql}"
    )
    assert params_log[0].get("namespace") == "tenant-acme"


@pytest.mark.unit
def test_v3_list_memories_filters_on_namespace_column():
    """v3 list_memories keeps namespace = %(namespace)s."""
    sqls: list = []
    params_log: list = []
    backend = _build_v3_backend_stub(sqls, params_log)

    backend.list_memories(namespace="legacy-tenant")

    sql = sqls[0]
    assert "namespace = %(namespace)s" in sql
    assert "metadata->>" not in sql
