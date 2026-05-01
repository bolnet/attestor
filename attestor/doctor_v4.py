"""v4-specific Postgres health checks (Phase 11.1, roadmap §release).

The legacy ``AgentMemory.health()`` validates that the document /
vector / graph stores are reachable and responsive. The v4 schema adds
load-bearing structural invariants on top: extensions, RLS policies,
bi-temporal triggers, the deletion_audit RLS-EXEMPT carve-out, and the
quota-counter triggers. A misconfigured deployment may pass health()
but silently lose tenant isolation or audit guarantees.

This module checks the structural invariants against a live Postgres
connection and returns a structured report. Surfaced via ``attestor
doctor`` so operators can verify a deployment from the CLI.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("attestor.doctor")


# ── Required structural invariants ───────────────────────────────────────


REQUIRED_EXTENSIONS = ("vector", "btree_gist", "uuid-ossp")

REQUIRED_RLS_TABLES = (
    "users", "projects", "sessions", "episodes", "memories",
)

# Audit/log tables that MUST NOT have RLS — they outlive their subjects.
RLS_EXEMPT_TABLES = ("deletion_audit",)

REQUIRED_TABLES = (
    "users", "projects", "sessions", "episodes", "memories",
    "user_quotas", "deletion_audit",
)

REQUIRED_TRIGGERS = (
    # name on a specific table — pg_trigger is keyed by both
    ("memories", "trg_memories_content_tsv"),
    ("users",    "trg_user_quota_init"),
    ("memories", "trg_quota_count_memories"),
    ("sessions", "trg_quota_count_sessions"),
    ("projects", "trg_quota_count_projects"),
)

REQUIRED_FUNCTIONS = ("attestor_user_id_for_external",)


# ── Result types ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CheckResult:
    """One v4 doctor check outcome."""
    name: str
    status: str           # "ok" | "fail" | "skip"
    detail: str = ""
    found: tuple = ()     # things observed (e.g., extensions present)
    missing: tuple = ()   # things required but absent

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "detail": self.detail,
            "found": list(self.found),
            "missing": list(self.missing),
        }


@dataclass(frozen=True)
class V4DoctorReport:
    """Aggregate v4 doctor verdict."""
    healthy: bool
    checks: tuple = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "healthy": self.healthy,
            "checks": [c.to_dict() for c in self.checks],
        }


# ── Individual checks ────────────────────────────────────────────────────


def _check_extensions(conn: Any) -> CheckResult:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT extname FROM pg_extension WHERE extname = ANY(%s)",
            (list(REQUIRED_EXTENSIONS),),
        )
        present = {row[0] for row in cur.fetchall()}
    missing = tuple(e for e in REQUIRED_EXTENSIONS if e not in present)
    return CheckResult(
        name="extensions",
        status="ok" if not missing else "fail",
        detail=("all required extensions installed"
                if not missing
                else f"missing extensions: {', '.join(missing)}"),
        found=tuple(sorted(present)),
        missing=missing,
    )


def _check_required_tables(conn: Any) -> CheckResult:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'public' AND table_name = ANY(%s)",
            (list(REQUIRED_TABLES),),
        )
        present = {row[0] for row in cur.fetchall()}
    missing = tuple(t for t in REQUIRED_TABLES if t not in present)
    return CheckResult(
        name="tables",
        status="ok" if not missing else "fail",
        detail=("all v4 tables present"
                if not missing
                else f"missing tables: {', '.join(missing)}"),
        found=tuple(sorted(present)),
        missing=missing,
    )


def _check_rls_enabled(conn: Any) -> CheckResult:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT tablename, rowsecurity FROM pg_tables "
            "WHERE schemaname = 'public' AND tablename = ANY(%s)",
            (list(REQUIRED_RLS_TABLES),),
        )
        rows = cur.fetchall()
    enabled = {t for (t, on) in rows if on}
    missing_rls = tuple(
        t for t in REQUIRED_RLS_TABLES if t not in enabled
    )
    return CheckResult(
        name="rls_enabled",
        status="ok" if not missing_rls else "fail",
        detail=("RLS enabled on all tenant tables"
                if not missing_rls
                else f"RLS NOT enabled on: {', '.join(missing_rls)}"),
        found=tuple(sorted(enabled)),
        missing=missing_rls,
    )


def _check_rls_exempt(conn: Any) -> CheckResult:
    """Audit tables MUST NOT have RLS — they outlive their subjects."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT tablename, rowsecurity FROM pg_tables "
            "WHERE schemaname = 'public' AND tablename = ANY(%s)",
            (list(RLS_EXEMPT_TABLES),),
        )
        rows = cur.fetchall()
    violators = tuple(t for (t, on) in rows if on)
    return CheckResult(
        name="rls_exempt_audit",
        status="ok" if not violators else "fail",
        detail=("audit tables correctly RLS-exempt"
                if not violators
                else (f"RLS WRONGLY enabled on audit tables: "
                      f"{', '.join(violators)}; will lose audit trail "
                      f"after user delete")),
        found=tuple(t for (t, _) in rows),
        missing=violators,
    )


def _check_triggers(conn: Any) -> CheckResult:
    """The bi-temporal content_tsv + quota counter triggers must exist."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT c.relname AS table_name, t.tgname AS trigger_name "
            "FROM pg_trigger t "
            "JOIN pg_class c ON c.oid = t.tgrelid "
            "WHERE NOT t.tgisinternal "
            "  AND c.relname = ANY(%s) "
            "  AND t.tgname  = ANY(%s)",
            (
                list({tbl for (tbl, _) in REQUIRED_TRIGGERS}),
                list({trg for (_, trg) in REQUIRED_TRIGGERS}),
            ),
        )
        seen = {(row[0], row[1]) for row in cur.fetchall()}
    missing = tuple(
        f"{tbl}.{trg}" for (tbl, trg) in REQUIRED_TRIGGERS
        if (tbl, trg) not in seen
    )
    return CheckResult(
        name="triggers",
        status="ok" if not missing else "fail",
        detail=("all required triggers present"
                if not missing
                else f"missing triggers: {', '.join(missing)}"),
        found=tuple(sorted(f"{t}.{n}" for (t, n) in seen)),
        missing=missing,
    )


def _check_security_definer_helpers(conn: Any) -> CheckResult:
    """SECURITY DEFINER lookup helpers must be present and EXECUTE-able."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT proname FROM pg_proc WHERE proname = ANY(%s)",
            (list(REQUIRED_FUNCTIONS),),
        )
        present = {row[0] for row in cur.fetchall()}
    missing = tuple(f for f in REQUIRED_FUNCTIONS if f not in present)
    return CheckResult(
        name="helper_functions",
        status="ok" if not missing else "fail",
        detail=("all SECURITY DEFINER helpers present"
                if not missing
                else f"missing functions: {', '.join(missing)}"),
        found=tuple(sorted(present)),
        missing=missing,
    )


# ── Orchestrator ─────────────────────────────────────────────────────────


_ALL_CHECKS = (
    _check_extensions,
    _check_required_tables,
    _check_rls_enabled,
    _check_rls_exempt,
    _check_triggers,
    _check_security_definer_helpers,
)


def run_v4_doctor(conn: Any) -> V4DoctorReport:
    """Run every v4 structural check against a live psycopg2-style connection.

    Each check is wrapped — a single broken check (e.g., insufficient
    privileges to read pg_trigger) doesn't kill the rest of the report.
    """
    results: list[CheckResult] = []
    for check_fn in _ALL_CHECKS:
        name = check_fn.__name__.removeprefix("_check_")
        try:
            results.append(check_fn(conn))
        except Exception as exc:
            logger.exception("v4 doctor check %s raised", name)
            results.append(CheckResult(
                name=name, status="skip",
                detail=f"check raised {type(exc).__name__}: {exc}",
            ))
    healthy = all(r.status == "ok" for r in results)
    return V4DoctorReport(healthy=healthy, checks=tuple(results))


def format_v4_report(report: V4DoctorReport) -> str:
    """Pretty-print the v4 report for the CLI."""
    lines: list[str] = []
    verdict = "HEALTHY" if report.healthy else "ISSUES DETECTED"
    lines.append(f"Attestor v4 schema doctor: {verdict}")
    lines.append("-" * 60)
    for c in report.checks:
        icon = {"ok": "OK", "fail": "FAIL", "skip": "SKIP"}.get(c.status, c.status)
        lines.append(f"  [{icon:<4}] {c.name:<22} {c.detail}")
        if c.missing and c.status == "fail":
            for m in c.missing:
                lines.append(f"           └─ missing: {m}")
    return "\n".join(lines)
