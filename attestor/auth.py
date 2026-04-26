"""HOSTED mode JWT auth middleware (Phase 8.4, roadmap §F).

Verifies an incoming JWT, extracts the ``sub`` claim as the external_id,
provisions the user on first contact (idempotent via UserRepo.create_or_get),
and sets ``request.state.user_id`` for downstream handlers.

In HOSTED mode, every request without a valid bearer token returns 401.
SOLO mode skips this middleware entirely (no auth required).
SHARED mode requires a token but doesn't enforce per-user quotas.

Verification:
  - audience  — config["jwt"]["audience"] must match aud claim
  - issuer    — config["jwt"]["issuer"]   must match iss claim (optional)
  - expiry    — exp claim must be in the future
  - signature — verified against config["jwt"]["public_key"] (HS256/RS256/etc.)

Threat model:
  - Wrong key / wrong audience / expired / missing sub → 401
  - Replay within TTL is allowed (callers should use short TTLs);
    revocation is the application's responsibility.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from attestor.mode import AttestorMode

logger = logging.getLogger("attestor.auth")


class AuthError(Exception):
    """Raised by the verifier for any reason a token is rejected."""


def _ensure_jwt():
    try:
        import jwt
    except ImportError as e:
        raise RuntimeError(
            "JWT auth requires the `pyjwt` package. "
            "Install via `pip install attestor[hosted]` or `pip install pyjwt`."
        ) from e
    return jwt


def verify_token(
    token: str,
    *,
    public_key: str,
    audience: Optional[str] = None,
    issuer: Optional[str] = None,
    algorithms: Optional[list[str]] = None,
    leeway_seconds: int = 0,
) -> Dict[str, Any]:
    """Decode + verify a JWT. Returns the claims dict.

    Raises AuthError on any verification failure with the reason string.
    """
    jwt = _ensure_jwt()
    if not token or not isinstance(token, str):
        raise AuthError("missing token")
    try:
        claims = jwt.decode(
            token, public_key,
            algorithms=algorithms or ["HS256", "RS256"],
            audience=audience,
            issuer=issuer,
            leeway=leeway_seconds,
            options={
                "require": ["sub", "exp"],
                "verify_signature": True,
                "verify_exp": True,
                "verify_aud": audience is not None,
                "verify_iss": issuer is not None,
            },
        )
    except jwt.ExpiredSignatureError as e:
        raise AuthError(f"expired: {e}") from e
    except jwt.InvalidAudienceError as e:
        raise AuthError(f"invalid audience: {e}") from e
    except jwt.InvalidIssuerError as e:
        raise AuthError(f"invalid issuer: {e}") from e
    except jwt.MissingRequiredClaimError as e:
        raise AuthError(f"missing claim: {e}") from e
    except jwt.InvalidTokenError as e:
        raise AuthError(f"invalid token: {e}") from e

    if not claims.get("sub"):
        raise AuthError("missing sub claim")
    return claims


def _extract_bearer(request: Request) -> Optional[str]:
    """Pull the bearer token from the Authorization header."""
    header = request.headers.get("authorization", "")
    if not header.lower().startswith("bearer "):
        return None
    return header[7:].strip() or None


class JWTAuthMiddleware(BaseHTTPMiddleware):
    """Starlette middleware that verifies a bearer JWT.

    Skips:
      - SOLO mode (no auth required by definition)
      - paths in ``unauthenticated_paths`` (default: /health)
    """

    SKIP_DEFAULT_PATHS = {"/health"}

    def __init__(
        self,
        app: Any,
        *,
        mode: AttestorMode,
        public_key: Optional[str] = None,
        audience: Optional[str] = None,
        issuer: Optional[str] = None,
        algorithms: Optional[list[str]] = None,
        leeway_seconds: int = 0,
        unauthenticated_paths: Optional[set[str]] = None,
        ensure_user_fn: Optional[Any] = None,
    ) -> None:
        super().__init__(app)
        self._mode = mode
        self._public_key = public_key
        self._audience = audience
        self._issuer = issuer
        self._algorithms = algorithms
        self._leeway = leeway_seconds
        self._skip = unauthenticated_paths or self.SKIP_DEFAULT_PATHS
        # Injected so tests can use a stub instead of an AgentMemory
        self._ensure_user = ensure_user_fn

    async def dispatch(self, request: Request, call_next: Any):
        # SOLO mode: no auth, just pass through.
        if self._mode is AttestorMode.SOLO:
            return await call_next(request)

        # Skip whitelisted paths (health checks, metrics, etc.)
        if request.url.path in self._skip:
            return await call_next(request)

        token = _extract_bearer(request)
        if token is None:
            return JSONResponse(
                {"ok": False, "error": "missing bearer token"},
                status_code=401,
            )
        if self._public_key is None:
            return JSONResponse(
                {"ok": False, "error": "auth not configured"},
                status_code=500,
            )
        try:
            claims = verify_token(
                token,
                public_key=self._public_key,
                audience=self._audience,
                issuer=self._issuer,
                algorithms=self._algorithms,
                leeway_seconds=self._leeway,
            )
        except AuthError as e:
            return JSONResponse(
                {"ok": False, "error": str(e)}, status_code=401,
            )

        external_id = claims["sub"]
        request.state.external_id = external_id
        request.state.jwt_claims = claims

        # Lazy first-time provisioning: ensure the user exists (idempotent).
        if self._ensure_user is not None:
            try:
                user = self._ensure_user(
                    external_id=external_id,
                    email=claims.get("email"),
                    display_name=claims.get("name") or claims.get("preferred_username"),
                )
                request.state.user_id = user.id
            except Exception as e:
                logger.warning("user provisioning failed for sub=%r: %s",
                               external_id, e)
                return JSONResponse(
                    {"ok": False, "error": "user provisioning failed"},
                    status_code=500,
                )

        return await call_next(request)
