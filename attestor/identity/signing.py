"""Ed25519 provenance signing for memory writes (Phase 8.1, roadmap §F).

Optional. When enabled in config, every memory inserted via the v4 path
gets an Ed25519 signature over the canonical tuple:

    sign_payload = id || agent_id || t_created || content_hash

The signature lands in ``memories.signature`` (column already exists).
Verification re-derives the payload from the row and checks the
signature against a public key.

Threat model:
  - DEFENDS against an attacker with raw DB write access who tries to
    forge or modify a memory: signature won't verify.
  - DOES NOT defend against the legitimate signing key being stolen.
    Rotate keys regularly; keep the secret in a secret manager.
  - DOES NOT defend against ordering attacks (re-arranging valid rows
    to imply a different sequence). Use bi-temporal columns + audit log
    for ordering integrity.

Why Ed25519: small keys (32 bytes), small signatures (64 bytes),
deterministic, fast, no nonce/padding pitfalls. Standard choice for
audit-trail signing.

Opt-in via config:
    {"signing": {"enabled": True,
                 "secret_key": "<base64-32-bytes>",
                 "public_key": "<base64-32-bytes>"}}

If `enabled=True` and no key set, AgentMemory generates an ephemeral
keypair at boot — useful for tests, NOT for production audit trails
(verifying offline requires the key to outlive the process).
"""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional, Tuple

logger = logging.getLogger("attestor.identity.signing")


def _ensure_crypto():
    """Lazy import — only callers who enable signing pay the import cost."""
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric import ed25519
    except ImportError as e:
        raise RuntimeError(
            "Ed25519 signing requires the `cryptography` package. "
            "Install via `pip install attestor[signing]` or `pip install cryptography`."
        ) from e
    return ed25519, InvalidSignature


@dataclass(frozen=True)
class SignatureKeypair:
    """Base64-encoded Ed25519 keypair for serialization to config / env vars."""
    secret_key_b64: str
    public_key_b64: str

    @classmethod
    def generate(cls) -> SignatureKeypair:
        ed25519, _ = _ensure_crypto()
        sk = ed25519.Ed25519PrivateKey.generate()
        from cryptography.hazmat.primitives import serialization
        sk_bytes = sk.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption(),
        )
        pk_bytes = sk.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        return cls(
            secret_key_b64=base64.b64encode(sk_bytes).decode("ascii"),
            public_key_b64=base64.b64encode(pk_bytes).decode("ascii"),
        )

    def secret_key_bytes(self) -> bytes:
        return base64.b64decode(self.secret_key_b64)

    def public_key_bytes(self) -> bytes:
        return base64.b64decode(self.public_key_b64)


# ──────────────────────────────────────────────────────────────────────────
# Canonical payload + sign / verify
# ──────────────────────────────────────────────────────────────────────────


def canonical_payload(
    *,
    memory_id: str,
    agent_id: Optional[str],
    t_created: Optional[Any],          # str | datetime | None
    content_hash: Optional[str],
) -> bytes:
    """Build the byte string that gets signed.

    Format: ``b"v1|<id>|<agent_id>|<t_created_iso>|<content_hash>"``.
    Stable across processes / Postgres timezones (always UTC ISO).

    None fields render as empty between separators so the format is
    unambiguous even when some fields are missing.
    """
    if isinstance(t_created, datetime):
        ts_str = t_created.isoformat()
    else:
        ts_str = str(t_created) if t_created is not None else ""
    parts = [
        "v1",
        str(memory_id) if memory_id is not None else "",
        agent_id or "",
        ts_str,
        content_hash or "",
    ]
    return "|".join(parts).encode("utf-8")


def sign_payload(payload: bytes, *, secret_key_bytes: bytes) -> str:
    """Sign and return base64 signature."""
    ed25519, _ = _ensure_crypto()
    sk = ed25519.Ed25519PrivateKey.from_private_bytes(secret_key_bytes)
    sig = sk.sign(payload)
    return base64.b64encode(sig).decode("ascii")


def verify_payload(
    payload: bytes,
    signature_b64: str,
    *,
    public_key_bytes: bytes,
) -> bool:
    """Verify; return True if signature valid, False otherwise."""
    ed25519, InvalidSignature = _ensure_crypto()
    try:
        sig = base64.b64decode(signature_b64)
    except Exception:
        return False
    pk = ed25519.Ed25519PublicKey.from_public_bytes(public_key_bytes)
    try:
        pk.verify(sig, payload)
        return True
    except InvalidSignature:
        return False


# ──────────────────────────────────────────────────────────────────────────
# High-level Memory helpers
# ──────────────────────────────────────────────────────────────────────────


def sign_memory(memory: Any, *, secret_key_bytes: bytes) -> str:
    """Compute the canonical payload from a Memory and sign it.

    Returns the base64 signature. Caller assigns it to memory.signature.
    """
    payload = canonical_payload(
        memory_id=memory.id,
        agent_id=memory.agent_id,
        t_created=memory.t_created,
        content_hash=memory.content_hash,
    )
    return sign_payload(payload, secret_key_bytes=secret_key_bytes)


def verify_memory(memory: Any, *, public_key_bytes: bytes) -> bool:
    """Verify the signature on a Memory row. Missing signature → False
    (an unsigned row can't be verified — caller decides whether that's
    OK)."""
    if not memory.signature:
        return False
    payload = canonical_payload(
        memory_id=memory.id,
        agent_id=memory.agent_id,
        t_created=memory.t_created,
        content_hash=memory.content_hash,
    )
    return verify_payload(
        payload, memory.signature, public_key_bytes=public_key_bytes,
    )


# ──────────────────────────────────────────────────────────────────────────
# Config-driven signer (used by AgentMemory)
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class Signer:
    """Wraps a keypair + the sign/verify operations for AgentMemory.

    Constructed from the ``signing`` block of MemoryConfig:

      {"enabled": True, "secret_key": "<b64>", "public_key": "<b64>"}

    If enabled=True and no keys, generates an ephemeral keypair (test-
    only). Production callers must persist their keys.
    """
    secret_key_bytes: bytes
    public_key_bytes: bytes
    enabled: bool = True

    @classmethod
    def from_config(cls, cfg: Optional[dict]) -> Optional[Signer]:
        if not cfg or not cfg.get("enabled"):
            return None
        sk_b64 = cfg.get("secret_key")
        pk_b64 = cfg.get("public_key")
        if sk_b64 and pk_b64:
            return cls(
                secret_key_bytes=base64.b64decode(sk_b64),
                public_key_bytes=base64.b64decode(pk_b64),
            )
        # Ephemeral keypair (tests + smoke tests only)
        kp = SignatureKeypair.generate()
        logger.warning(
            "signing enabled with no keypair in config; generated an "
            "EPHEMERAL keypair — NOT suitable for production audit"
        )
        return cls(
            secret_key_bytes=kp.secret_key_bytes(),
            public_key_bytes=kp.public_key_bytes(),
        )

    def sign(self, memory: Any) -> str:
        return sign_memory(memory, secret_key_bytes=self.secret_key_bytes)

    def verify(self, memory: Any) -> bool:
        return verify_memory(memory, public_key_bytes=self.public_key_bytes)
