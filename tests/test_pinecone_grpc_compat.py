# SPDX-FileCopyrightText: 2026 Surendra Singh <66422685+bolnet@users.noreply.github.com>
# SPDX-License-Identifier: MIT
"""``bind_local_grpc_index`` works on both pinecone 8.x and 9.x SDKs.

Bug under test (found in live e2e 2026-09-03): the local-emulator data
plane imported ``pinecone.grpc.GrpcIndex`` (9.x only) while pyproject
allows ``pinecone>=5`` and poetry.lock pins 8.1.2, whose class is
``GRPCIndex`` + ``GRPCClientConfig``. Result: ``attestor doctor`` reported
the vector store "Not initialized (cannot import name 'GrpcIndex')".
"""
from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from attestor.store._pinecone_grpc import bind_local_grpc_index

pytestmark = pytest.mark.unit


class _Pc:
    def __init__(self) -> None:
        self.index_calls: list[dict[str, Any]] = []

    def Index(self, **kwargs: Any) -> str:  # noqa: N802 - SDK name
        self.index_calls.append(kwargs)
        return "index-8x"


def _install_fake_grpc(monkeypatch: pytest.MonkeyPatch, module: types.ModuleType) -> None:
    monkeypatch.setitem(sys.modules, "pinecone.grpc", module)


def test_uses_grpcindex_kwargs_on_9x(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    def grpc_index(**kwargs: Any) -> str:
        calls.append(kwargs)
        return "index-9x"

    mod = types.ModuleType("pinecone.grpc")
    mod.GrpcIndex = grpc_index  # type: ignore[attr-defined]
    _install_fake_grpc(monkeypatch, mod)
    pc = _Pc()

    out = bind_local_grpc_index(pc, index_name="mem", host="localhost:5081", api_key="pclocal")

    assert out == "index-9x"
    assert calls == [{"host": "localhost:5081", "api_key": "pclocal", "secure": False}]
    assert pc.index_calls == []


def test_falls_back_to_grpcclientconfig_on_8x(monkeypatch: pytest.MonkeyPatch) -> None:
    class GRPCClientConfig:  # noqa: N801 - SDK name
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    mod = types.ModuleType("pinecone.grpc")
    mod.GRPCClientConfig = GRPCClientConfig  # type: ignore[attr-defined]
    _install_fake_grpc(monkeypatch, mod)
    pc = _Pc()

    out = bind_local_grpc_index(pc, index_name="mem", host="localhost:5081", api_key="pclocal")

    assert out == "index-8x"
    (call,) = pc.index_calls
    assert call["name"] == "mem"
    assert call["host"] == "localhost:5081"
    assert call["grpc_config"].kwargs == {"secure": False}


def test_neither_api_raises_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_grpc(monkeypatch, types.ModuleType("pinecone.grpc"))
    with pytest.raises(ImportError, match="pinecone.grpc"):
        bind_local_grpc_index(_Pc(), index_name="mem", host="h", api_key="k")
