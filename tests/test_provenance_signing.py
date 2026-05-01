"""Phase 8.1 — Ed25519 provenance signing tests.

Pure crypto tests run without DB. End-to-end tests verify
AgentMemory.add() signs and AgentMemory.verify_memory() round-trips
against the live v4 schema.
"""

from __future__ import annotations

import base64
import os
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

try:
    import psycopg2
    HAVE_PSYCOPG2 = True
except ImportError:
    HAVE_PSYCOPG2 = False

from attestor.identity.signing import (
    SignatureKeypair,
    Signer,
    canonical_payload,
    sign_memory,
    sign_payload,
    verify_memory,
    verify_payload,
)
from attestor.models import Memory


# ──────────────────────────────────────────────────────────────────────────
# Keypair
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_generate_keypair_produces_32_byte_keys() -> None:
    kp = SignatureKeypair.generate()
    assert len(kp.secret_key_bytes()) == 32
    assert len(kp.public_key_bytes()) == 32


@pytest.mark.unit
def test_keypair_b64_round_trips() -> None:
    kp = SignatureKeypair.generate()
    sk2 = base64.b64decode(kp.secret_key_b64)
    pk2 = base64.b64decode(kp.public_key_b64)
    assert sk2 == kp.secret_key_bytes()
    assert pk2 == kp.public_key_bytes()


# ──────────────────────────────────────────────────────────────────────────
# Canonical payload
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_canonical_payload_format() -> None:
    out = canonical_payload(
        memory_id="m-1", agent_id="planner",
        t_created="2026-04-26T10:00:00+00:00",
        content_hash="abc123",
    )
    assert out == b"v1|m-1|planner|2026-04-26T10:00:00+00:00|abc123"


@pytest.mark.unit
def test_canonical_payload_handles_none_fields() -> None:
    """Missing fields render as empty between separators."""
    out = canonical_payload(
        memory_id="m-1", agent_id=None, t_created=None, content_hash=None,
    )
    assert out == b"v1|m-1|||"


@pytest.mark.unit
def test_canonical_payload_accepts_datetime_t_created() -> None:
    dt = datetime(2026, 4, 26, 10, 0, 0, tzinfo=timezone.utc)
    out = canonical_payload(
        memory_id="m-1", agent_id="x", t_created=dt, content_hash="h",
    )
    assert b"2026-04-26T10:00:00+00:00" in out


@pytest.mark.unit
def test_canonical_payload_distinct_for_distinct_inputs() -> None:
    a = canonical_payload(memory_id="m-1", agent_id="x",
                          t_created="t", content_hash="h")
    b = canonical_payload(memory_id="m-2", agent_id="x",
                          t_created="t", content_hash="h")
    c = canonical_payload(memory_id="m-1", agent_id="y",
                          t_created="t", content_hash="h")
    d = canonical_payload(memory_id="m-1", agent_id="x",
                          t_created="t", content_hash="other")
    assert len({a, b, c, d}) == 4


# ──────────────────────────────────────────────────────────────────────────
# sign + verify round trip
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_sign_verify_round_trip() -> None:
    kp = SignatureKeypair.generate()
    payload = b"v1|m-1|planner|2026|h"
    sig = sign_payload(payload, secret_key_bytes=kp.secret_key_bytes())
    assert verify_payload(payload, sig, public_key_bytes=kp.public_key_bytes())


@pytest.mark.unit
def test_verify_rejects_modified_payload() -> None:
    kp = SignatureKeypair.generate()
    sig = sign_payload(b"original", secret_key_bytes=kp.secret_key_bytes())
    assert not verify_payload(
        b"tampered", sig, public_key_bytes=kp.public_key_bytes(),
    )


@pytest.mark.unit
def test_verify_rejects_wrong_public_key() -> None:
    kp_a = SignatureKeypair.generate()
    kp_b = SignatureKeypair.generate()
    sig = sign_payload(b"x", secret_key_bytes=kp_a.secret_key_bytes())
    assert not verify_payload(
        b"x", sig, public_key_bytes=kp_b.public_key_bytes(),
    )


@pytest.mark.unit
def test_verify_rejects_garbage_signature() -> None:
    kp = SignatureKeypair.generate()
    assert not verify_payload(
        b"x", "not-base64!!", public_key_bytes=kp.public_key_bytes(),
    )


@pytest.mark.unit
def test_signature_is_64_bytes_b64_encoded() -> None:
    """Ed25519 produces 64-byte signatures → 88-char base64 (with padding)."""
    kp = SignatureKeypair.generate()
    sig = sign_payload(b"x", secret_key_bytes=kp.secret_key_bytes())
    raw = base64.b64decode(sig)
    assert len(raw) == 64


# ──────────────────────────────────────────────────────────────────────────
# sign_memory / verify_memory
# ──────────────────────────────────────────────────────────────────────────


def _mem(content="hi", agent_id="planner", t_created="t-1", chash="h"):
    return Memory(
        id="m-1", content=content, content_hash=chash,
        agent_id=agent_id, t_created=t_created,
    )


@pytest.mark.unit
def test_sign_memory_attaches_to_signature_field() -> None:
    kp = SignatureKeypair.generate()
    m = _mem()
    m = replace(m, signature=sign_memory(m, secret_key_bytes=kp.secret_key_bytes()))
    assert m.signature
    assert verify_memory(m, public_key_bytes=kp.public_key_bytes())


@pytest.mark.unit
def test_verify_memory_returns_false_when_unsigned() -> None:
    kp = SignatureKeypair.generate()
    m = _mem()  # no signature set
    assert not verify_memory(m, public_key_bytes=kp.public_key_bytes())


@pytest.mark.unit
def test_verify_memory_rejects_tampered_content() -> None:
    """Modifying content_hash post-signing must fail verification."""
    kp = SignatureKeypair.generate()
    m = _mem()
    m = replace(m, signature=sign_memory(m, secret_key_bytes=kp.secret_key_bytes()))
    m = replace(m, content_hash="tampered-hash")  # attacker swaps content
    assert not verify_memory(m, public_key_bytes=kp.public_key_bytes())


@pytest.mark.unit
def test_verify_memory_rejects_swapped_agent_id() -> None:
    """Forging agent_id (claiming a different writer) must fail verification."""
    kp = SignatureKeypair.generate()
    m = _mem()
    m = replace(m, signature=sign_memory(m, secret_key_bytes=kp.secret_key_bytes()))
    m = replace(m, agent_id="evil-agent")
    assert not verify_memory(m, public_key_bytes=kp.public_key_bytes())


# ──────────────────────────────────────────────────────────────────────────
# Signer (config-driven)
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_signer_from_config_disabled_returns_none() -> None:
    assert Signer.from_config(None) is None
    assert Signer.from_config({"enabled": False}) is None
    assert Signer.from_config({}) is None


@pytest.mark.unit
def test_signer_from_config_with_keys() -> None:
    kp = SignatureKeypair.generate()
    s = Signer.from_config({
        "enabled": True,
        "secret_key": kp.secret_key_b64,
        "public_key": kp.public_key_b64,
    })
    assert s is not None
    assert s.secret_key_bytes == kp.secret_key_bytes()


@pytest.mark.unit
def test_signer_ephemeral_when_keys_absent_but_enabled() -> None:
    """Tests / smoke runs get an ephemeral keypair."""
    s = Signer.from_config({"enabled": True})
    assert s is not None
    # Round-trips with itself
    m = _mem()
    m = replace(m, signature=s.sign(m))
    assert s.verify(m) is True


@pytest.mark.unit
def test_signer_round_trip_via_class_methods() -> None:
    kp = SignatureKeypair.generate()
    s = Signer(
        secret_key_bytes=kp.secret_key_bytes(),
        public_key_bytes=kp.public_key_bytes(),
    )
    m = _mem()
    m = replace(m, signature=s.sign(m))
    assert s.verify(m)


# ──────────────────────────────────────────────────────────────────────────
# AgentMemory end-to-end (live)
# ──────────────────────────────────────────────────────────────────────────


PG_URL = os.environ.get(
    "PG_TEST_URL",
    "postgresql://postgres:attestor@localhost:5432/attestor_v4_test",
)
SCHEMA_PATH = Path(__file__).resolve().parent.parent / "attestor" / "store" / "schema.sql"


def _reachable() -> bool:
    if not HAVE_PSYCOPG2:
        return False
    try:
        c = psycopg2.connect(PG_URL, connect_timeout=2)
        c.close()
        return True
    except Exception:
        return False


def _ollama_up() -> bool:
    try:
        import requests
        r = requests.get("http://localhost:11434/api/tags", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


@pytest.fixture
def fresh_schema():
    if not _reachable():
        pytest.skip("local Postgres unreachable")
    c = psycopg2.connect(PG_URL)
    c.autocommit = True
    raw = SCHEMA_PATH.read_text().replace("{embedding_dim}", "1024")
    with c.cursor() as cur:
        for tbl in ("memories", "episodes", "sessions", "projects", "users"):
            cur.execute(f"DROP TABLE IF EXISTS {tbl} CASCADE")
        cur.execute(raw)
    c.close()
    yield


@pytest.mark.live
@pytest.mark.skipif(not _ollama_up(), reason="Ollama not running")
def test_agent_memory_signs_on_add(fresh_schema, tmp_path) -> None:
    from attestor.core import AgentMemory
    kp = SignatureKeypair.generate()
    mem = AgentMemory(tmp_path, config={
        "mode": "solo",
        "backends": ["postgres"],
        "backend_configs": {"postgres": {
            "url": "postgresql://localhost:5432",
            "database": "attestor_v4_test",
            "auth": {"username": "postgres", "password": "attestor"},
            "v4": True, "skip_schema_init": True,
        }},
        "signing": {
            "enabled": True,
            "secret_key": kp.secret_key_b64,
            "public_key": kp.public_key_b64,
        },
    })
    try:
        assert mem.signing_enabled
        m = mem.add("user prefers dark mode", category="preference",
                    entity="UI")
        assert m.signature, "memory was not signed"
        assert mem.verify_memory(m.id) is True
    finally:
        mem.close()


@pytest.mark.live
@pytest.mark.skipif(not _ollama_up(), reason="Ollama not running")
def test_agent_memory_detects_tampered_row(fresh_schema, tmp_path) -> None:
    """Direct SQL UPDATE that modifies content WITHOUT re-signing must fail
    verification — the audit trail catches DB-level tampering."""
    from attestor.core import AgentMemory
    kp = SignatureKeypair.generate()
    mem = AgentMemory(tmp_path, config={
        "mode": "solo",
        "backends": ["postgres"],
        "backend_configs": {"postgres": {
            "url": "postgresql://localhost:5432",
            "database": "attestor_v4_test",
            "auth": {"username": "postgres", "password": "attestor"},
            "v4": True, "skip_schema_init": True,
        }},
        "signing": {
            "enabled": True,
            "secret_key": kp.secret_key_b64,
            "public_key": kp.public_key_b64,
        },
    })
    try:
        m = mem.add("legitimate fact", category="preference", entity="x")
        assert mem.verify_memory(m.id) is True

        # Attacker bypasses the app and modifies the row directly
        admin = psycopg2.connect(PG_URL); admin.autocommit = True
        with admin.cursor() as cur:
            cur.execute(
                "UPDATE memories SET content_hash = %s WHERE id = %s",
                ("tampered-hash", m.id),
            )
        admin.close()
        assert mem.verify_memory(m.id) is False
    finally:
        mem.close()


@pytest.mark.live
def test_verify_memory_raises_when_signing_disabled(fresh_schema, tmp_path) -> None:
    """Calling verify_memory without the signer configured is a programmer
    error — the public key needs to be in the same place."""
    from attestor.core import AgentMemory
    if not _ollama_up():
        pytest.skip("Ollama not running")
    mem = AgentMemory(tmp_path, config={
        "mode": "solo",
        "backends": ["postgres"],
        "backend_configs": {"postgres": {
            "url": "postgresql://localhost:5432",
            "database": "attestor_v4_test",
            "auth": {"username": "postgres", "password": "attestor"},
            "v4": True, "skip_schema_init": True,
        }},
    })
    try:
        assert not mem.signing_enabled
        with pytest.raises(RuntimeError, match="signing"):
            mem.verify_memory("any-id")
    finally:
        mem.close()
