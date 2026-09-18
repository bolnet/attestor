"""Governance activity failures are persisted in Temporal history — redact them.

Review finding: ``describe_error(exc)`` put ``str(exc)`` verbatim into the
``ApplicationError`` message, and driver exceptions (psycopg2, Pinecone,
Neo4j) can echo DSNs / credentials. The message that reaches workflow
history is now sanitised and capped, and the raw exception is no longer
chained as ``__cause__`` (Temporal serialises the cause chain too).
"""

from __future__ import annotations

import logging

import pytest

pytest.importorskip("temporalio", reason="temporalio not installed (attestor[durable])")

from temporalio.exceptions import ApplicationError  # noqa: E402

from attestor.durable.activities._errors import raise_for  # noqa: E402
from attestor.durable.activities._sanitize import (  # noqa: E402
    MAX_ERROR_MESSAGE_CHARS,
    REDACTED,
    sanitize_error_message,
)

pytestmark = pytest.mark.unit


def test_plain_messages_pass_through_unchanged():
    assert sanitize_error_message("RuntimeError: pg down") == "RuntimeError: pg down"


def test_credentialed_urls_are_redacted():
    msg = (
        "OperationalError: could not connect to postgresql://app:pw-secret@db.internal:5432/att "
        "(bolt://neo4j:graph-secret@graph:7687 too)"
    )
    out = sanitize_error_message(msg)
    assert "pw-secret" not in out
    assert "graph-secret" not in out
    assert "OperationalError" in out
    assert REDACTED in out


def test_key_value_secrets_are_redacted():
    msg = (
        "OperationalError: host=db password=hunter2 user=app; api_key=pcsk_abc123 "
        "token: tok-xyz Authorization: Bearer eyJabc.def"
    )
    out = sanitize_error_message(msg)
    for leaked in ("hunter2", "pcsk_abc123", "tok-xyz", "eyJabc.def"):
        assert leaked not in out, out
    assert "host=db" in out
    assert "user=app" in out


def test_bare_api_key_shapes_are_redacted():
    out = sanitize_error_message("AuthError: key sk-proj-ABCDEFGH12345678 rejected (401)")
    assert "ABCDEFGH12345678" not in out
    assert "401" in out


def test_messages_are_capped():
    out = sanitize_error_message("x" * (MAX_ERROR_MESSAGE_CHARS * 3))
    assert len(out) <= MAX_ERROR_MESSAGE_CHARS + len("…")


def test_whitespace_is_collapsed():
    assert sanitize_error_message("a\n\n   b\t c") == "a b c"


def test_raise_for_sanitises_and_drops_the_cause_chain():
    exc = RuntimeError("dsn postgresql://app:pw-secret@db/att refused")
    with pytest.raises(ApplicationError) as info:
        raise_for(
            exc, subject="user=u-1", step="forget_doc",
            transient_type="ForgetFailed", permanent_type="ForgetPermanent",
            logger=logging.getLogger("test"),
        )
    assert "pw-secret" not in str(info.value)
    assert "RuntimeError" in str(info.value)
    assert info.value.__cause__ is None
    assert info.value.type == "ForgetFailed"
    assert not info.value.non_retryable


def test_raise_for_never_logs_the_secret(caplog):
    exc = RuntimeError("password=hunter2 rejected")
    with caplog.at_level(logging.DEBUG), pytest.raises(ApplicationError):
        raise_for(
            exc, subject="s", step="st", transient_type="T", permanent_type="P",
            logger=logging.getLogger("test.sanitize"),
        )
    assert "hunter2" not in caplog.text


def test_derive_activity_sanitises_and_drops_the_cause_chain():
    from attestor.durable.activities.derive import DeriveActivities
    from attestor.durable.models import DeriveRef

    class _Mem:
        def derive_vector(self, memory_id, *, user_id=None, agent_id=None):  # noqa: ANN001
            raise RuntimeError("pinecone https://user:pw-secret@host/x said no")

    acts = DeriveActivities(lambda: _Mem())
    with pytest.raises(ApplicationError) as info:
        acts.embed_and_upsert(DeriveRef(memory_id="m-1", user_id="u-1"))
    assert "pw-secret" not in str(info.value)
    assert info.value.__cause__ is None
    assert not info.value.non_retryable
