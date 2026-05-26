# SPDX-License-Identifier: MIT
"""Live, env-gated proof of cross-store per-project isolation.

Requires the canonical stack (Postgres + Pinecone + Neo4j). Gated behind
`POSTGRES_URL` (same convention as the rest of the live suite; see conftest).
Do NOT run during a benchmark — it writes to the live DB.
"""

import os
from pathlib import Path

import pytest

from attestor.core import AgentMemory
from attestor.hooks._tenant import resolve_tenant

# Same gate + credential wiring as the rest of the live suite: backend creds
# come from POSTGRES_URL via the conftest ``test_config`` fixture (a bare
# AgentMemory(path) has no password and fails with "no password supplied").
pytestmark = pytest.mark.skipif(
    not os.environ.get("POSTGRES_URL"),
    reason="requires a live Postgres backend (set POSTGRES_URL) + Pinecone + Neo4j",
)


def test_two_projects_do_not_share_memory(tmp_path: Path):
    # Build the canonical v4 stack config from configs/attestor.yaml — same
    # recipe the benches use. The shared DB already has the v4 schema, so skip
    # re-init. (A bare AgentMemory(path) or the minimal conftest test_config
    # runs the v3 schema path and collides with the v4 columns.)
    from attestor.config import build_backend_config, get_stack

    backend_config = build_backend_config(get_stack())
    backend_config["backend_configs"]["postgres"]["skip_schema_init"] = True
    store = str(tmp_path / "store")
    mem = AgentMemory(store, config=backend_config)
    try:
        proj_a = tmp_path / "alpha"
        (proj_a / ".git").mkdir(parents=True)
        proj_b = tmp_path / "beta"
        (proj_b / ".git").mkdir(parents=True)

        ua, ns_a = resolve_tenant(mem, str(proj_a))
        ub, ns_b = resolve_tenant(mem, str(proj_b))
        assert ua != ub

        written = mem.add(
            "Alpha uses Rust for the parser",
            tags=["decision"], category="project",
            user_id=ua, namespace=ns_a, scope="project",
        )
        try:
            # Recall from project B must NOT see project A's memory.
            hits_b = mem.recall(
                "what language for the parser", user_id=ub, namespace=ns_b,
            )
            assert all("Alpha uses Rust" not in r.memory.content for r in hits_b)

            # Recall from project A DOES see it.
            hits_a = mem.recall(
                "what language for the parser", user_id=ua, namespace=ns_a,
            )
            assert any("Alpha uses Rust" in r.memory.content for r in hits_a)
        finally:
            # Don't accumulate rows across repeated live runs.
            mem.forget(written.id)
    finally:
        mem.close()
