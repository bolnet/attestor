"""Phase 8.4 — JWT auth middleware tests.

Pure unit tests for verify_token + middleware behavior. Uses HS256 with
a static secret key so we don't need a live KMS / IDP.
"""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import MagicMock

import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

import jwt as pyjwt

from attestor.auth import AuthError, JWTAuthMiddleware, verify_token
from attestor.mode import AttestorMode


SECRET = "test-secret-32-bytes-min-or-the-key-too-short-warning"


# ──────────────────────────────────────────────────────────────────────────
# verify_token — pure function tests
# ──────────────────────────────────────────────────────────────────────────


def _make_token(**overrides: Any) -> str:
    payload = {
        "sub": "user-42",
        "exp": int(time.time()) + 3600,
        "aud": "attestor",
        "iss": "https://idp.example.com",
        **overrides,
    }
    return pyjwt.encode(payload, SECRET, algorithm="HS256")


@pytest.mark.unit
def test_verify_token_happy_path() -> None:
    tok = _make_token()
    claims = verify_token(
        tok, public_key=SECRET, audience="attestor",
        issuer="https://idp.example.com",
    )
    assert claims["sub"] == "user-42"


@pytest.mark.unit
def test_verify_token_rejects_missing_token() -> None:
    with pytest.raises(AuthError, match="missing token"):
        verify_token("", public_key=SECRET)


@pytest.mark.unit
def test_verify_token_rejects_expired() -> None:
    tok = _make_token(exp=int(time.time()) - 60)
    with pytest.raises(AuthError, match="expired"):
        verify_token(tok, public_key=SECRET, audience="attestor")


@pytest.mark.unit
def test_verify_token_rejects_wrong_audience() -> None:
    tok = _make_token(aud="other-service")
    with pytest.raises(AuthError, match="audience"):
        verify_token(tok, public_key=SECRET, audience="attestor")


@pytest.mark.unit
def test_verify_token_rejects_wrong_issuer() -> None:
    tok = _make_token(iss="https://evil.example.com")
    with pytest.raises(AuthError, match="issuer"):
        verify_token(
            tok, public_key=SECRET, audience="attestor",
            issuer="https://idp.example.com",
        )


@pytest.mark.unit
def test_verify_token_rejects_wrong_signature() -> None:
    tok = _make_token()
    with pytest.raises(AuthError, match="invalid token"):
        verify_token(tok, public_key="wrong-secret", audience="attestor")


@pytest.mark.unit
def test_verify_token_rejects_missing_sub_claim() -> None:
    """A token without sub must be rejected (we use sub as external_id)."""
    payload = {
        "exp": int(time.time()) + 3600,
        "aud": "attestor",
    }
    tok = pyjwt.encode(payload, SECRET, algorithm="HS256")
    with pytest.raises(AuthError, match="sub"):
        verify_token(tok, public_key=SECRET, audience="attestor")


@pytest.mark.unit
def test_verify_token_audience_optional() -> None:
    """When audience=None, the aud claim is not enforced."""
    tok = _make_token(aud="anything")
    claims = verify_token(tok, public_key=SECRET, audience=None)
    assert claims["sub"] == "user-42"


# ──────────────────────────────────────────────────────────────────────────
# Middleware — Starlette integration via TestClient
# ──────────────────────────────────────────────────────────────────────────


def _make_app(
    *,
    mode: AttestorMode,
    ensure_user: Any = None,
    public_key: str = SECRET,
    audience: str = "attestor",
) -> Starlette:
    async def echo(request: Request) -> JSONResponse:
        body = {
            "ok": True,
            "user_id": getattr(request.state, "user_id", None),
            "external_id": getattr(request.state, "external_id", None),
        }
        return JSONResponse(body)

    async def health(request: Request) -> JSONResponse:
        return JSONResponse({"ok": True, "kind": "health"})

    routes = [
        Route("/echo", echo, methods=["GET"]),
        Route("/health", health, methods=["GET"]),
    ]
    app = Starlette(routes=routes)
    app.add_middleware(
        JWTAuthMiddleware,
        mode=mode,
        public_key=public_key,
        audience=audience,
        ensure_user_fn=ensure_user,
    )
    return app


@pytest.mark.unit
def test_middleware_solo_mode_passes_through() -> None:
    """SOLO mode: no auth required. Token absent → request still succeeds."""
    app = _make_app(mode=AttestorMode.SOLO)
    client = TestClient(app)
    r = client.get("/echo")
    assert r.status_code == 200
    assert r.json()["ok"] is True


@pytest.mark.unit
def test_middleware_hosted_mode_rejects_missing_bearer() -> None:
    app = _make_app(mode=AttestorMode.HOSTED)
    client = TestClient(app)
    r = client.get("/echo")
    assert r.status_code == 401
    assert "bearer" in r.json()["error"].lower()


@pytest.mark.unit
def test_middleware_hosted_mode_health_skips_auth() -> None:
    """Health check must respond even without a token (probes don't auth)."""
    app = _make_app(mode=AttestorMode.HOSTED)
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200


@pytest.mark.unit
def test_middleware_hosted_mode_accepts_valid_token() -> None:
    fake_user = MagicMock()
    fake_user.id = "user-uuid-123"
    ensure_user = MagicMock(return_value=fake_user)

    app = _make_app(mode=AttestorMode.HOSTED, ensure_user=ensure_user)
    client = TestClient(app)
    tok = _make_token(sub="alice@example.com", email="alice@example.com",
                      name="Alice")
    r = client.get("/echo", headers={"Authorization": f"Bearer {tok}"})
    assert r.status_code == 200
    body = r.json()
    assert body["external_id"] == "alice@example.com"
    assert body["user_id"] == "user-uuid-123"

    # ensure_user called with parsed claims
    args = ensure_user.call_args
    assert args.kwargs["external_id"] == "alice@example.com"
    assert args.kwargs["email"] == "alice@example.com"
    assert args.kwargs["display_name"] == "Alice"


@pytest.mark.unit
def test_middleware_rejects_expired_token() -> None:
    app = _make_app(mode=AttestorMode.HOSTED, ensure_user=MagicMock())
    client = TestClient(app)
    tok = _make_token(exp=int(time.time()) - 60)
    r = client.get("/echo", headers={"Authorization": f"Bearer {tok}"})
    assert r.status_code == 401
    assert "expired" in r.json()["error"].lower()


@pytest.mark.unit
def test_middleware_rejects_wrong_audience() -> None:
    app = _make_app(mode=AttestorMode.HOSTED, ensure_user=MagicMock())
    client = TestClient(app)
    tok = _make_token(aud="other-svc")
    r = client.get("/echo", headers={"Authorization": f"Bearer {tok}"})
    assert r.status_code == 401
    assert "audience" in r.json()["error"].lower()


@pytest.mark.unit
def test_middleware_rejects_when_public_key_unconfigured() -> None:
    """HOSTED mode without public_key set is a deployment error → 500."""
    app = _make_app(mode=AttestorMode.HOSTED, public_key=None,
                    ensure_user=MagicMock())
    client = TestClient(app)
    tok = _make_token()
    r = client.get("/echo", headers={"Authorization": f"Bearer {tok}"})
    assert r.status_code == 500


@pytest.mark.unit
def test_middleware_provisioning_failure_returns_500() -> None:
    """A DB error during ensure_user shouldn't expose internals."""
    bad = MagicMock(side_effect=RuntimeError("db on fire"))
    app = _make_app(mode=AttestorMode.HOSTED, ensure_user=bad)
    client = TestClient(app)
    tok = _make_token()
    r = client.get("/echo", headers={"Authorization": f"Bearer {tok}"})
    assert r.status_code == 500
    assert "user provisioning failed" in r.json()["error"]
