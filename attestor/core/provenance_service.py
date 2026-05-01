"""Provenance signing mixin for AgentMemory (split from core.py).

Exposes the public verification surface (``verify_memory`` /
``signing_enabled``). The actual signing call site sits inline in
``agent_memory.py``'s ``add()`` because it participates in the post-insert
UPDATE round-trip.

Mixin assumes the composing class wires up ``self._signer`` (Optional
``Signer``) and ``self._store`` in ``__init__``.
"""

from __future__ import annotations


class _ProvenanceMixin:
    """Mixin holding the public provenance surface for AgentMemory."""

    # -- v4 provenance signing (Phase 8.1) --

    def verify_memory(self, memory_id: str) -> bool:
        """Verify the Ed25519 signature on a stored memory.

        Returns True if signed AND the signature matches; False if
        unsigned, missing, or tampered.

        Raises RuntimeError if signing isn't configured for this
        instance — verification needs the public key in the same place.
        """
        if self._signer is None:
            raise RuntimeError(
                "verify_memory requires config['signing'] to be set "
                "(public_key needed for verification)"
            )
        row = self._store.get(memory_id)
        if row is None:
            return False
        return self._signer.verify(row)

    @property
    def signing_enabled(self) -> bool:
        return self._signer is not None
