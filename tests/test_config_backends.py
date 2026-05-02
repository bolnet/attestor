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
            "backends": ["postgres", "pinecone", "neo4j"],
            "postgres": {"url": "postgresql://localhost:5432/attestor"},
        })
        assert cfg.backends == ["postgres", "pinecone", "neo4j"]
        assert cfg.backend_configs["postgres"]["url"] == \
            "postgresql://localhost:5432/attestor"

    def test_env_var_not_resolved_at_config_time(self):
        os.environ["TEST_PG_PW"] = "secret123"
        cfg = MemoryConfig.from_dict({
            "backends": ["postgres"],
            "postgres": {"url": "postgresql://localhost:5432", "password": "$TEST_PG_PW"},
        })
        assert cfg.backend_configs["postgres"]["password"] == "$TEST_PG_PW"
        del os.environ["TEST_PG_PW"]

    def test_roundtrip_save_load(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = MemoryConfig(
                backends=["postgres", "pinecone", "neo4j"],
                backend_configs={
                    "postgres": {"url": "postgresql://localhost:5432"},
                },
            )
            save_config(Path(d), cfg)
            loaded = load_config(Path(d))
            assert loaded.backends == ["postgres", "pinecone", "neo4j"]
            assert loaded.backend_configs["postgres"]["url"] == \
                "postgresql://localhost:5432"

    def test_default_config_unchanged(self):
        cfg = MemoryConfig.from_dict({"default_token_budget": 5000})
        assert cfg.backends == ["postgres", "neo4j"]
        assert cfg.default_token_budget == 5000
