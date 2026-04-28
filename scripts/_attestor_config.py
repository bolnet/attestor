"""Back-compat shim — the canonical loader lives in ``attestor.config``.

This file used to be the loader. It was promoted into the package
itself so production code (CLI, api server, benchmark modules) can
import it. We keep this shim so existing scripts that still import
from ``scripts._attestor_config`` keep working.

New code should import from ``attestor.config`` directly:

    from attestor.config import (
        get_stack,
        load_stack,
        StackConfig,
        configure_embedder,
        build_backend_config,
        verify_neo4j_reachable,
        confirm_or_exit,
        print_stack_banner,
        cloud_target,
        DEFAULT_CONFIG,
    )
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make sure the repo root (and therefore the `attestor` package) is on
# sys.path when this module is imported as `scripts._attestor_config`.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from attestor.config import (  # noqa: E402,F401
    DEFAULT_CONFIG,
    CloudTarget,
    EmbedderCfg,
    ImageCfg,
    ModelsCfg,
    Neo4jCfg,
    PostgresCfg,
    StackConfig,
    build_backend_config,
    cloud_target,
    configure_embedder,
    confirm_or_exit,
    get_stack,
    load_stack,
    print_stack_banner,
    reset_stack,
    set_stack,
    verify_neo4j_reachable,
)
