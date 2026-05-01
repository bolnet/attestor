"""Phase 11.1 — v4 doctor tests.

Two layers:
  - Pure unit tests with a stub psycopg2 cursor. Verify each individual
    check classifies "missing" / "present" correctly.
  - A live test (gated by PG_TEST_URL) that runs against the real
    schema and confirms a fresh apply passes every check.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from attestor.doctor_v4 import (
    REQUIRED_EXTENSIONS, REQUIRED_RLS_TABLES, REQUIRED_TABLES,
    REQUIRED_TRIGGERS, RLS_EXEMPT_TABLES,
    CheckResult, V4DoctorReport, format_v4_report, run_v4_doctor,
)


# ── Stub cursor (no live DB) ─────────────────────────────────────────────


class _StubCursor:
    """psycopg2-style cursor returning canned rows per query."""

    def __init__(self, responses: list[tuple[str, list[tuple]]]) -> None:
        # responses: list of (query_substring, rows) pairs, consumed in order.
        # Test asserts the calls happen in a known sequence.
        self._responses = list(responses)
        self._last_rows: list[tuple] = []
        self.queries: list[tuple[str, Any]] = []

    def execute(self, query: str, params=None) -> None:
        self.queries.append((query, params))
        if not self._responses:
            self._last_rows = []
            return
        substr, rows = self._responses.pop(0)
        # Loose match — let tests pin to a fragment of the query so
        # rewriting whitespace doesn't break tests.
        assert substr in query, (
            f"expected query containing {substr!r}; got: {query[:200]}"
        )
        self._last_rows = rows

    def fetchall(self) -> list[tuple]:
        return list(self._last_rows)

    def fetchone(self):
        return self._last_rows[0] if self._last_rows else None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _StubConn:
    def __init__(self, responses: list[tuple[str, list[tuple]]]) -> None:
        self._responses = responses

    def cursor(self):
        return _StubCursor(self._responses)


def _conn_for_full_run(*,
                       extensions=REQUIRED_EXTENSIONS,
                       tables=REQUIRED_TABLES,
                       rls_on=REQUIRED_RLS_TABLES,
                       audit_rls_on=False,
                       triggers=REQUIRED_TRIGGERS,
                       functions=("attestor_user_id_for_external",)) -> Any:
    """Build a stub connection that returns sensible rows for every check
    in order. Override individual sections to simulate misconfiguration."""

    # Each check gets its OWN cursor (we use `with conn.cursor() as cur`).
    # _StubConn returns a fresh cursor per call, but the cursor only sees
    # the responses we give it. The simplest way to handle that: have one
    # response set per check and a counter that picks the right list.
    response_lists = [
        # extensions
        [("FROM pg_extension", [(e,) for e in extensions])],
        # tables
        [("FROM information_schema.tables", [(t,) for t in tables])],
        # rls_enabled
        [("FROM pg_tables", [(t, t in rls_on) for t in REQUIRED_RLS_TABLES])],
        # rls_exempt
        [("FROM pg_tables", [(t, audit_rls_on) for t in RLS_EXEMPT_TABLES])],
        # triggers
        [("FROM pg_trigger", [(tbl, trg) for (tbl, trg) in triggers])],
        # helper_functions
        [("FROM pg_proc", [(f,) for f in functions])],
    ]

    class _SequentialConn:
        def __init__(self) -> None:
            self._idx = 0

        def cursor(self):
            cur = _StubCursor(response_lists[self._idx])
            self._idx += 1
            return cur

    return _SequentialConn()


# ── Individual check assertions via run_v4_doctor ────────────────────────


@pytest.mark.unit
def test_v4_doctor_all_present_is_healthy() -> None:
    rep = run_v4_doctor(_conn_for_full_run())
    assert rep.healthy is True
    assert all(c.status == "ok" for c in rep.checks)


@pytest.mark.unit
def test_v4_doctor_flags_missing_extension() -> None:
    rep = run_v4_doctor(_conn_for_full_run(extensions=("vector", "btree_gist")))
    ext_check = next(c for c in rep.checks if c.name == "extensions")
    assert ext_check.status == "fail"
    assert "uuid-ossp" in ext_check.missing
    assert rep.healthy is False


@pytest.mark.unit
def test_v4_doctor_flags_missing_table() -> None:
    rep = run_v4_doctor(_conn_for_full_run(
        tables=tuple(t for t in REQUIRED_TABLES if t != "user_quotas"),
    ))
    tbl_check = next(c for c in rep.checks if c.name == "tables")
    assert tbl_check.status == "fail"
    assert "user_quotas" in tbl_check.missing


@pytest.mark.unit
def test_v4_doctor_flags_rls_disabled_on_tenant_table() -> None:
    rep = run_v4_doctor(_conn_for_full_run(
        rls_on=tuple(t for t in REQUIRED_RLS_TABLES if t != "memories"),
    ))
    rls = next(c for c in rep.checks if c.name == "rls_enabled")
    assert rls.status == "fail"
    assert "memories" in rls.missing


@pytest.mark.unit
def test_v4_doctor_flags_rls_wrongly_enabled_on_audit() -> None:
    """deletion_audit MUST NOT have RLS — the row outlives its subject."""
    rep = run_v4_doctor(_conn_for_full_run(audit_rls_on=True))
    audit = next(c for c in rep.checks if c.name == "rls_exempt_audit")
    assert audit.status == "fail"
    assert "deletion_audit" in audit.missing
    assert "lose audit trail" in audit.detail


@pytest.mark.unit
def test_v4_doctor_flags_missing_trigger() -> None:
    """Missing the content_tsv trigger means BM25 lane is silently dead."""
    rep = run_v4_doctor(_conn_for_full_run(
        triggers=tuple(
            (tbl, trg) for (tbl, trg) in REQUIRED_TRIGGERS
            if trg != "trg_memories_content_tsv"
        ),
    ))
    trg = next(c for c in rep.checks if c.name == "triggers")
    assert trg.status == "fail"
    assert any("trg_memories_content_tsv" in m for m in trg.missing)


@pytest.mark.unit
def test_v4_doctor_flags_missing_helper_function() -> None:
    rep = run_v4_doctor(_conn_for_full_run(functions=()))
    fn = next(c for c in rep.checks if c.name == "helper_functions")
    assert fn.status == "fail"
    assert "attestor_user_id_for_external" in fn.missing


# ── Resilience: a check that raises shouldn't kill the report ────────────


@pytest.mark.unit
def test_v4_doctor_isolates_check_failure() -> None:
    """If pg_trigger isn't readable (insufficient privs), other checks
    still run and surface their own results."""

    class _PartiallyBrokenConn:
        def __init__(self) -> None:
            self._idx = 0

        def cursor(self):
            response_lists = [
                # extensions OK
                [("FROM pg_extension", [(e,) for e in REQUIRED_EXTENSIONS])],
                # tables OK
                [("FROM information_schema.tables",
                  [(t,) for t in REQUIRED_TABLES])],
                # rls_enabled OK
                [("FROM pg_tables",
                  [(t, True) for t in REQUIRED_RLS_TABLES])],
                # rls_exempt OK
                [("FROM pg_tables",
                  [(t, False) for t in RLS_EXEMPT_TABLES])],
            ]
            if self._idx < len(response_lists):
                cur = _StubCursor(response_lists[self._idx])
                self._idx += 1
                return cur

            # triggers + helper_functions checks — raise
            class _Raises:
                def __enter__(self): return self
                def __exit__(self, *a): return False
                def execute(self, *a, **k):
                    raise RuntimeError("permission denied for pg_catalog")
                def fetchall(self): return []
                def fetchone(self): return None
            return _Raises()

    rep = run_v4_doctor(_PartiallyBrokenConn())
    # 4 OK + 2 SKIP — we still get a useful report instead of a crash
    statuses = [c.status for c in rep.checks]
    assert statuses.count("ok") == 4
    assert statuses.count("skip") == 2
    assert rep.healthy is False  # skips count as not-healthy


# ── format_v4_report ─────────────────────────────────────────────────────


@pytest.mark.unit
def test_format_v4_report_healthy_says_so() -> None:
    rep = V4DoctorReport(
        healthy=True,
        checks=(CheckResult(name="extensions", status="ok",
                            detail="all installed"),),
    )
    txt = format_v4_report(rep)
    assert "HEALTHY" in txt
    assert "OK" in txt


@pytest.mark.unit
def test_format_v4_report_lists_missing_items() -> None:
    rep = V4DoctorReport(
        healthy=False,
        checks=(CheckResult(
            name="extensions", status="fail",
            detail="missing extensions: uuid-ossp",
            missing=("uuid-ossp",),
        ),),
    )
    txt = format_v4_report(rep)
    assert "ISSUES DETECTED" in txt
    assert "FAIL" in txt
    assert "uuid-ossp" in txt


# ── Live integration test (gated) ────────────────────────────────────────


PG_URL = os.environ.get(
    "PG_TEST_URL",
    "postgresql://postgres:attestor@localhost:5432/attestor_v4_test",
)
SCHEMA_PATH = (Path(__file__).resolve().parent.parent
               / "attestor" / "store" / "schema.sql")


def _pg_reachable() -> bool:
    try:
        import psycopg2
    except ImportError:
        return False
    try:
        c = psycopg2.connect(PG_URL, connect_timeout=2)
        c.close()
        return True
    except Exception:
        return False


@pytest.mark.live
@pytest.mark.skipif(not _pg_reachable(), reason="local Postgres unreachable")
def test_v4_doctor_passes_against_fresh_schema() -> None:
    """The full v4 schema, freshly applied, must pass every doctor check."""
    import psycopg2
    raw = SCHEMA_PATH.read_text().replace("{embedding_dim}", "16")

    with psycopg2.connect(PG_URL) as setup, setup.cursor() as cur:
        for tbl in ("deletion_audit", "user_quotas", "memories", "episodes",
                    "sessions", "projects", "users"):
            cur.execute(f"DROP TABLE IF EXISTS {tbl} CASCADE")
        cur.execute(raw)
        setup.commit()

    with psycopg2.connect(PG_URL) as conn:
        report = run_v4_doctor(conn)

    if not report.healthy:
        # Surface the failure detail so debugging is easy
        print(format_v4_report(report))
    assert report.healthy is True
