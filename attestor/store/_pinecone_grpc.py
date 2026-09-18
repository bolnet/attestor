# SPDX-FileCopyrightText: 2026 Surendra Singh <66422685+bolnet@users.noreply.github.com>
# SPDX-License-Identifier: MIT
"""Bind the Pinecone *Local* gRPC data plane across SDK generations.

pyproject allows ``pinecone>=5``. The insecure (http) gRPC index the
local emulator needs is constructed differently per major version:

* 9.x: ``pinecone.grpc.GrpcIndex(host=, api_key=, secure=False)``
  (``GRPCClientConfig`` was removed and ``PineconeGRPC.Index`` rejects
  ``secure=``).
* 5.x-8.x: ``PineconeGRPC.Index(name=, host=, grpc_config=GRPCClientConfig(secure=False))``.

Kept in its own module so the backend stays within its size budget and
the version probe is unit-testable with a faked ``pinecone.grpc``.
"""

from __future__ import annotations

import importlib
from typing import Any


def _grpc_module() -> Any:
    return importlib.import_module("pinecone.grpc")


def bind_local_grpc_index(pc: Any, *, index_name: str, host: str, api_key: str) -> Any:
    """Return a gRPC index bound to ``host`` over plain http (emulator transport)."""
    grpc = _grpc_module()
    grpc_index_cls = getattr(grpc, "GrpcIndex", None)
    if grpc_index_cls is not None:
        return grpc_index_cls(host=host, api_key=api_key, secure=False)
    client_config_cls = getattr(grpc, "GRPCClientConfig", None)
    if client_config_cls is None:
        raise ImportError(
            "pinecone.grpc exposes neither GrpcIndex (pinecone>=9) nor "
            "GRPCClientConfig (pinecone<9); cannot bind the Pinecone Local index"
        )
    return pc.Index(name=index_name, host=host, grpc_config=client_config_cls(secure=False))
