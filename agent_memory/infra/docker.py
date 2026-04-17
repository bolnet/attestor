# agent_memory/infra/docker.py
"""Opt-in Docker auto-start for local development backends.

Requires ``pip install "memwright[docker]"``. Imported lazily from
:meth:`agent_memory.core.AgentMemory._ensure_docker` only when
``backend.docker = true`` in config.
"""
from __future__ import annotations

import logging
import subprocess
import time
from dataclasses import dataclass
from typing import Dict

from agent_memory.store._extras import require_extra

# Raises MissingExtraError at import time if the `docker` SDK is missing.
_docker = require_extra("docker", extra="docker")

logger = logging.getLogger("agent_memory.infra.docker")

_CONTAINER_PREFIX = "memwright-"
_START_TIMEOUT_SECONDS = 30
_START_POLL_INTERVAL_SECONDS = 1.0


@dataclass(frozen=True)
class ContainerInfo:
    """Immutable descriptor for a running container managed by Memwright."""

    name: str
    port: int


class DockerManager:
    """Minimal wrapper around the docker SDK + CLI.

    Construction never touches the daemon; the first daemon call happens in
    :meth:`ensure_running`. This keeps unit tests mock-friendly.
    """

    def __init__(self) -> None:
        self._client = None  # lazy: see _get_client

    # -- Public API ------------------------------------------------------

    @staticmethod
    def container_name(backend_name: str) -> str:
        return f"{_CONTAINER_PREFIX}{backend_name}"

    def ensure_running(
        self,
        backend_name: str,
        image: str,
        port: int,
        env: Dict[str, str],
    ) -> ContainerInfo:
        """Start the container if not running; return its :class:`ContainerInfo`."""
        if not self._is_running(backend_name):
            self._start_container(backend_name, image, port, env)
            if not self._wait_running(backend_name):
                raise RuntimeError(
                    f"container {self.container_name(backend_name)} did not start within "
                    f"{_START_TIMEOUT_SECONDS}s"
                )
        return ContainerInfo(name=self.container_name(backend_name), port=port)

    def stop(self, backend_name: str) -> None:
        name = self.container_name(backend_name)
        subprocess.run(["docker", "stop", name], check=False)

    def health_check(self, backend_name: str) -> bool:
        return self._is_running(backend_name)

    # -- Internals -------------------------------------------------------

    def _get_client(self):
        if self._client is None:
            self._client = _docker.from_env()
        return self._client

    def _is_running(self, backend_name: str) -> bool:
        name = self.container_name(backend_name)
        try:
            container = self._get_client().containers.get(name)
        except _docker.errors.NotFound:  # type: ignore[attr-defined]
            return False
        return container.status == "running"

    def _start_container(
        self,
        backend_name: str,
        image: str,
        port: int,
        env: Dict[str, str],
    ) -> None:
        name = self.container_name(backend_name)
        client = self._get_client()
        try:
            existing = client.containers.get(name)
            existing.start()
            return
        except _docker.errors.NotFound:  # type: ignore[attr-defined]
            pass
        client.containers.run(
            image,
            name=name,
            detach=True,
            ports={f"{port}/tcp": port},
            environment=env,
            restart_policy={"Name": "unless-stopped"},
        )
        logger.info("started container %s on port %d", name, port)

    def _wait_running(self, backend_name: str) -> bool:
        deadline = time.monotonic() + _START_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            if self._is_running(backend_name):
                return True
            time.sleep(_START_POLL_INTERVAL_SECONDS)
        return False
