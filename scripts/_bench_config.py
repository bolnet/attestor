"""Back-compat shim — the canonical loader lives in ``attestor.bench_config``.

Mirrors the pattern of ``scripts/_attestor_config.py``. New code should
import from ``attestor.bench_config`` directly.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from attestor.bench_config import (  # noqa: E402,F401
    DEFAULT_BENCH_CONFIG,
    DEFAULT_STACK_CONFIG,
    BenchCfg,
    KeyOverlapError,
    KnowledgeUpdatesCfg,
    LMECfg,
    ReportCfg,
    assert_disjoint_keys,
    get_bench,
    load_bench,
    print_bench_banner,
    reset_bench,
)
