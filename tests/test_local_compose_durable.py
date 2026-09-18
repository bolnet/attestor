# SPDX-FileCopyrightText: 2026 Surendra Singh <66422685+bolnet@users.noreply.github.com>
# SPDX-License-Identifier: MIT
"""The bundled local compose file: Temporal is an opt-in ``durable`` profile.

Guards docs/plans/temporal-integration.md §2 ("Quickstart stays
zero-question") — the three default role services carry NO profile, and
the Temporal server + UI are only ever started with ``--profile durable``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

COMPOSE = Path(__file__).resolve().parents[1] / "attestor/infra/local/docker-compose.yml"
DEFAULT_SERVICES = ("postgres", "neo4j", "pinecone", "attestor-api")

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def services() -> dict[str, Any]:
    return yaml.safe_load(COMPOSE.read_text())["services"]


def _env(service: dict[str, Any]) -> dict[str, str]:
    raw = service.get("environment", {})
    if isinstance(raw, list):
        return dict(item.split("=", 1) for item in raw)
    return {k: str(v) for k, v in raw.items()}


@pytest.mark.parametrize("name", DEFAULT_SERVICES)
def test_default_services_have_no_profile(services: dict[str, Any], name: str) -> None:
    assert "profiles" not in services[name], f"{name} must start without --profile"


def test_temporal_server_is_durable_profile_on_shared_postgres(services: dict[str, Any]) -> None:
    svc = services["temporal"]
    assert svc["container_name"] == "attestor_temporal_server"
    assert svc["image"].startswith("temporalio/auto-setup:")
    assert svc["profiles"] == ["durable"]
    env = _env(svc)
    assert env["DB"] == "postgres12"
    assert env["POSTGRES_SEEDS"] == "postgres"
    assert env["POSTGRES_USER"] == services["postgres"]["environment"]["POSTGRES_USER"]
    # Same password interpolation as the postgres service — one credential.
    assert env["POSTGRES_PWD"] == services["postgres"]["environment"]["POSTGRES_PASSWORD"]
    # Namespace matches configs/attestor.yaml durable.namespace.
    assert env["DEFAULT_NAMESPACE"] == "attestor"
    assert "7233:7233" in svc["ports"]
    assert svc["depends_on"]["postgres"]["condition"] == "service_healthy"


def test_temporal_ui_is_durable_profile_on_8233(services: dict[str, Any]) -> None:
    svc = services["temporal-ui"]
    assert svc["container_name"] == "attestor_temporal_ui"
    assert svc["image"].startswith("temporalio/ui:")
    assert svc["profiles"] == ["durable"]
    env = _env(svc)
    assert env["TEMPORAL_ADDRESS"] == "temporal:7233"
    assert env["TEMPORAL_DEFAULT_NAMESPACE"] == "attestor"
    assert "8233:8080" in svc["ports"]
    assert "temporal" in svc["depends_on"]
