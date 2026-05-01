# tests/test_docker.py
"""Tests for the optional Docker infrastructure manager.

These tests mock the docker SDK entirely; no daemon is required. Network-
touching integration tests live in ``test_docker_live.py`` (not created here).
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("docker", reason="install attestor[docker] extra to run these tests")

from attestor.infra.docker import ContainerInfo, DockerManager


class TestContainerInfo:
    def test_is_frozen(self) -> None:
        info = ContainerInfo(name="attestor-arangodb", port=8529)
        with pytest.raises(Exception):
            info.name = "other"  # type: ignore[misc]


class TestDockerManager:
    def test_container_name_prefix(self) -> None:
        assert DockerManager().container_name("arangodb") == "attestor-arangodb"

    def test_ensure_running_starts_container(self) -> None:
        dm = DockerManager()
        with patch.object(dm, "_is_running", return_value=False), \
             patch.object(dm, "_start_container") as mock_start, \
             patch.object(dm, "_wait_running", return_value=True):
            info = dm.ensure_running(
                backend_name="arangodb",
                image="arangodb:3.12",
                port=8529,
                env={"ARANGO_NO_AUTH": "1"},
            )
        mock_start.assert_called_once()
        assert info == ContainerInfo(name="attestor-arangodb", port=8529)

    def test_ensure_running_reuses_existing(self) -> None:
        dm = DockerManager()
        with patch.object(dm, "_is_running", return_value=True), \
             patch.object(dm, "_start_container") as mock_start:
            info = dm.ensure_running(
                backend_name="arangodb",
                image="arangodb:3.12",
                port=8529,
                env={},
            )
        mock_start.assert_not_called()
        assert info.name == "attestor-arangodb"

    def test_ensure_running_raises_when_start_times_out(self) -> None:
        dm = DockerManager()
        with patch.object(dm, "_is_running", return_value=False), \
             patch.object(dm, "_start_container"), \
             patch.object(dm, "_wait_running", return_value=False):
            with pytest.raises(RuntimeError, match="did not start within"):
                dm.ensure_running(
                    backend_name="arangodb",
                    image="arangodb:3.12",
                    port=8529,
                    env={},
                )

    def test_stop_container(self) -> None:
        dm = DockerManager()
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            dm.stop("arangodb")
        mock_run.assert_called()
        args, _ = mock_run.call_args
        assert "attestor-arangodb" in args[0]

    def test_health_check_reflects_running_state(self) -> None:
        dm = DockerManager()
        with patch.object(dm, "_is_running", return_value=True):
            assert dm.health_check("arangodb") is True
        with patch.object(dm, "_is_running", return_value=False):
            assert dm.health_check("arangodb") is False


class TestMissingExtra:
    def test_import_raises_when_docker_sdk_absent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """If the optional `docker` SDK is missing, importing the module must
        raise MissingExtraError pointing at `attestor[docker]`."""
        import importlib
        import sys

        from attestor.store._extras import MissingExtraError

        # Simulate `docker` SDK being absent
        monkeypatch.setitem(sys.modules, "docker", None)
        monkeypatch.delitem(sys.modules, "attestor.infra.docker", raising=False)

        with pytest.raises(MissingExtraError) as exc:
            importlib.import_module("attestor.infra.docker")
        assert "attestor[docker]" in str(exc.value)
