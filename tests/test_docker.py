"""Tests for Docker infrastructure manager."""

import subprocess
import pytest
from unittest.mock import patch, MagicMock
from agent_memory.infra.docker import DockerManager, ContainerInfo


class TestDockerManager:
    def test_container_name_prefix(self):
        dm = DockerManager()
        assert dm.container_name("arangodb") == "memwright-arangodb"

    def test_ensure_running_starts_container(self):
        dm = DockerManager()
        with patch.object(dm, "_is_running", return_value=False), \
             patch.object(dm, "_start_container") as mock_start, \
             patch.object(dm, "_wait_healthy", return_value=True):
            info = dm.ensure_running(
                backend_name="arangodb",
                image="arangodb/arangodb:latest",
                port=8529,
                env={"ARANGO_NO_AUTH": "1"},
            )
            mock_start.assert_called_once()
            assert info.name == "memwright-arangodb"
            assert info.port == 8529

    def test_ensure_running_reuses_existing(self):
        dm = DockerManager()
        with patch.object(dm, "_is_running", return_value=True), \
             patch.object(dm, "_start_container") as mock_start:
            info = dm.ensure_running(
                backend_name="arangodb",
                image="arangodb/arangodb:latest",
                port=8529,
                env={},
            )
            mock_start.assert_not_called()
            assert info.name == "memwright-arangodb"

    def test_stop_container(self):
        dm = DockerManager()
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            dm.stop("arangodb")
            mock_run.assert_called()

    def test_health_check_returns_bool(self):
        dm = DockerManager()
        with patch.object(dm, "_is_running", return_value=True):
            assert dm.health_check("arangodb") is True
        with patch.object(dm, "_is_running", return_value=False):
            assert dm.health_check("arangodb") is False
