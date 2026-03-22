"""Docker container manager for local backend provisioning."""

from __future__ import annotations

import logging
import subprocess
import time
from dataclasses import dataclass
from typing import Dict, Optional

logger = logging.getLogger("agent_memory")

CONTAINER_PREFIX = "memwright"


@dataclass(frozen=True)
class ContainerInfo:
    name: str
    port: int
    image: str


class DockerManager:
    """Auto-provision Docker containers for local backends.

    Uses docker CLI via subprocess -- no docker-py dependency.
    Containers persist across sessions (reusable).
    """

    def container_name(self, backend_name: str) -> str:
        return f"{CONTAINER_PREFIX}-{backend_name}"

    def ensure_running(
        self,
        backend_name: str,
        image: str,
        port: int,
        env: Dict[str, str],
        health_timeout: int = 30,
        container_port: Optional[int] = None,
    ) -> ContainerInfo:
        """Start container if not running. Returns container info."""
        name = self.container_name(backend_name)

        if not self._is_running(name):
            self._start_container(name, image, port, env, container_port=container_port)
            self._wait_healthy(name, timeout=health_timeout)

        return ContainerInfo(name=name, port=port, image=image)

    def stop(self, backend_name: str) -> None:
        """Stop and remove a managed container."""
        name = self.container_name(backend_name)
        subprocess.run(
            ["docker", "rm", "-f", name],
            capture_output=True,
            timeout=30,
        )
        logger.info("Stopped container %s", name)

    def health_check(self, backend_name: str) -> bool:
        """Check if a managed container is running."""
        name = self.container_name(backend_name)
        return self._is_running(name)

    def cleanup(self) -> None:
        """Stop all memwright containers."""
        result = subprocess.run(
            ["docker", "ps", "-a", "--filter", f"name={CONTAINER_PREFIX}-",
             "--format", "{{.Names}}"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            for name in result.stdout.strip().split("\n"):
                if name:
                    subprocess.run(
                        ["docker", "rm", "-f", name],
                        capture_output=True, timeout=30,
                    )

    def _is_running(self, name: str) -> bool:
        result = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", name],
            capture_output=True, text=True, timeout=10,
        )
        return result.returncode == 0 and result.stdout.strip() == "true"

    def _start_container(
        self,
        name: str,
        image: str,
        port: int,
        env: Dict[str, str],
    ) -> None:
        cmd = [
            "docker", "run", "-d",
            "--name", name,
            "-p", f"{port}:{port}",
        ]
        for k, v in env.items():
            cmd.extend(["-e", f"{k}={v}"])
        cmd.append(image)

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to start {name}: {result.stderr.strip()}"
            )
        logger.info("Started container %s from %s on port %d", name, image, port)

    def _wait_healthy(self, name: str, timeout: int = 30) -> bool:
        """Poll until container is running or timeout."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._is_running(name):
                return True
            time.sleep(1)
        raise TimeoutError(f"Container {name} not healthy after {timeout}s")
