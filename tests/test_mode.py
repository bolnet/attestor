"""Phase 2 chunk 1 — AttestorMode detection rules (defaults.md §1)."""

from __future__ import annotations

import pytest

from attestor.mode import AttestorMode, detect_mode


@pytest.mark.unit
def test_default_is_solo() -> None:
    """Empty env → SOLO. The natural default for fresh installs."""
    assert detect_mode(env={}) is AttestorMode.SOLO


@pytest.mark.unit
def test_explicit_mode_wins() -> None:
    for s, expected in [
        ("solo", AttestorMode.SOLO),
        ("hosted", AttestorMode.HOSTED),
        ("shared", AttestorMode.SHARED),
        ("SOLO", AttestorMode.SOLO),
        ("Hosted", AttestorMode.HOSTED),
    ]:
        assert detect_mode(env={"ATTESTOR_MODE": s}) is expected


@pytest.mark.unit
def test_invalid_explicit_falls_back_to_solo() -> None:
    """A typo in ATTESTOR_MODE shouldn't crash the library on import."""
    assert detect_mode(env={"ATTESTOR_MODE": "borked"}) is AttestorMode.SOLO


@pytest.mark.unit
def test_auth_provider_implies_hosted() -> None:
    for p in ("auth0", "clerk", "cognito", "oidc"):
        assert detect_mode(env={"ATTESTOR_AUTH_PROVIDER": p}) is AttestorMode.HOSTED


@pytest.mark.unit
def test_require_auth_implies_shared() -> None:
    for v in ("1", "true", "True", "yes"):
        assert detect_mode(env={"ATTESTOR_REQUIRE_AUTH": v}) is AttestorMode.SHARED


@pytest.mark.unit
def test_explicit_overrides_auth_provider() -> None:
    """If both are set, ATTESTOR_MODE wins (explicit > inferred)."""
    env = {"ATTESTOR_MODE": "solo", "ATTESTOR_AUTH_PROVIDER": "auth0"}
    assert detect_mode(env=env) is AttestorMode.SOLO


@pytest.mark.unit
def test_auth_provider_beats_require_auth() -> None:
    """If both inferred signals fire, HOSTED wins (more specific intent)."""
    env = {"ATTESTOR_AUTH_PROVIDER": "clerk", "ATTESTOR_REQUIRE_AUTH": "true"}
    assert detect_mode(env=env) is AttestorMode.HOSTED


@pytest.mark.unit
def test_require_auth_falsy_falls_back_to_solo() -> None:
    """A literal `false` (or `0`) should not flip the mode."""
    for v in ("0", "false", "False", ""):
        assert detect_mode(env={"ATTESTOR_REQUIRE_AUTH": v}) is AttestorMode.SOLO
