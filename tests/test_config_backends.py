"""Tests for config with backend settings."""

import os
import tempfile
from pathlib import Path

from attestor.utils.config import MemoryConfig, load_config, save_config


class TestBackendConfig:
    def test_default_backends(self):
        cfg = MemoryConfig()
        assert cfg.backends == ["postgres", "neo4j"]

    def test_custom_backends(self):
        cfg = MemoryConfig.from_dict({
            "backends": ["arangodb"],
            "arangodb": {"mode": "local", "port": 8529},
        })
        assert cfg.backends == ["arangodb"]
        assert cfg.backend_configs["arangodb"]["mode"] == "local"

    def test_env_var_not_resolved_at_config_time(self):
        os.environ["TEST_ARANGO_PW"] = "secret123"
        cfg = MemoryConfig.from_dict({
            "backends": ["arangodb"],
            "arangodb": {"mode": "cloud", "password": "$TEST_ARANGO_PW"},
        })
        assert cfg.backend_configs["arangodb"]["password"] == "$TEST_ARANGO_PW"
        del os.environ["TEST_ARANGO_PW"]

    def test_roundtrip_save_load(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = MemoryConfig(
                backends=["arangodb"],
                backend_configs={"arangodb": {"mode": "local"}},
            )
            save_config(Path(d), cfg)
            loaded = load_config(Path(d))
            assert loaded.backends == ["arangodb"]
            assert loaded.backend_configs["arangodb"]["mode"] == "local"

    def test_default_config_unchanged(self):
        cfg = MemoryConfig.from_dict({"default_token_budget": 5000})
        assert cfg.backends == ["postgres", "neo4j"]
        assert cfg.default_token_budget == 5000
