# SPDX-FileCopyrightText: 2026 Surendra Singh <66422685+bolnet@users.noreply.github.com>
# SPDX-License-Identifier: MIT
"""Durable config resolution + the error types every durable entry point raises.

The YAML block is parsed in ``attestor.config`` (``StackConfig.durable``,
``DurableCfg``). This module layers the runtime concerns on top:

* ``TEMPORAL_*`` env passthrough (``resolve_durable``)
* explicit CLI overrides (``apply_overrides``)
* the fail-loudly gates (``require_enabled``, ``require_temporalio``)

No ``temporalio`` import at module level — plain installs may import
this to learn that durable is disabled.
"""

from __future__ import annotations

import os
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from attestor.config import get_stack

if TYPE_CHECKING:
    from collections.abc import Mapping
    from types import ModuleType

    from attestor.config import DurableCfg

INSTALL_HINT = 'pip install "attestor[durable]"'

ENV_ADDRESS = "TEMPORAL_ADDRESS"
ENV_NAMESPACE = "TEMPORAL_NAMESPACE"
ENV_TASK_QUEUE = "TEMPORAL_TASK_QUEUE"
ENV_TLS = "TEMPORAL_TLS"

_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"0", "false", "no", "off"})


class DurableError(RuntimeError):
    """Base class for every durable-layer failure."""


class DurableDisabledError(DurableError):
    """``durable.enabled`` is false — caller must stay in-process."""


class DurableUnavailableError(DurableError):
    """``temporalio`` missing or the Temporal server is unreachable."""


def _parse_bool(name: str, raw: str) -> bool:
    value = raw.strip().lower()
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    raise ValueError(
        f"{name}={raw!r} must be one of {sorted(_TRUE)} / {sorted(_FALSE)}"
    )


def _env_overrides(env: Mapping[str, str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if address := env.get(ENV_ADDRESS):
        out["address"] = address
    if namespace := env.get(ENV_NAMESPACE):
        out["namespace"] = namespace
    if task_queue := env.get(ENV_TASK_QUEUE):
        out["task_queue"] = task_queue
    if (tls := env.get(ENV_TLS)) is not None and tls != "":
        out["tls"] = _parse_bool(ENV_TLS, tls)
    return out


def apply_overrides(cfg: DurableCfg, overrides: Mapping[str, Any]) -> DurableCfg:
    """Return a copy of ``cfg`` with the non-``None`` overrides applied."""
    clean = {k: v for k, v in overrides.items() if v is not None}
    return replace(cfg, **clean) if clean else cfg


def resolve_durable(
    cfg: DurableCfg | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> DurableCfg:
    """YAML value (or ``cfg``) with ``TEMPORAL_*`` env overrides applied.

    Resolution order matches the rest of the config layer: explicit env
    var beats the YAML value. ``enabled`` is YAML-only — flipping durable
    on is a deliberate config edit, not an env accident.
    """
    base = cfg if cfg is not None else get_stack().durable
    return apply_overrides(base, _env_overrides(os.environ if env is None else env))


def require_enabled(cfg: DurableCfg) -> DurableCfg:
    """Return ``cfg`` or raise :class:`DurableDisabledError`."""
    if not cfg.enabled:
        raise DurableDisabledError(
            "durable.enabled is false in configs/attestor.yaml; "
            "set `durable.enabled: true` to use Temporal-backed governance jobs"
        )
    return cfg


def require_temporalio() -> ModuleType:
    """Import ``temporalio`` or raise :class:`DurableUnavailableError`."""
    try:
        import temporalio
    except ImportError as exc:
        raise DurableUnavailableError(
            f"temporalio is not installed ({exc}); run {INSTALL_HINT}"
        ) from exc
    return temporalio
