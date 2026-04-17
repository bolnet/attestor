import builtins
import sys
import pytest
from unittest.mock import patch


def _patch_missing(names_to_fail):
    real_import = builtins.__import__
    def fake_import(name, *args, **kwargs):
        if name in names_to_fail or any(name.startswith(n + ".") for n in names_to_fail):
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)
    return fake_import


def test_arango_missing_raises_actionable_error():
    """When python-arango is missing, re-import surfaces memwright[arangodb] hint."""
    sys.modules.pop("agent_memory.store.arango_backend", None)
    sys.modules.pop("arango", None)
    with patch("builtins.__import__", _patch_missing({"arango"})):
        with pytest.raises(ImportError) as exc_info:
            import agent_memory.store.arango_backend  # noqa: F401
    assert "memwright[arangodb]" in str(exc_info.value)


def test_postgres_missing_raises_actionable_error():
    """When psycopg2 is missing, re-import surfaces memwright[postgres] hint."""
    sys.modules.pop("agent_memory.store.postgres_backend", None)
    sys.modules.pop("psycopg2", None)
    sys.modules.pop("psycopg2.extras", None)
    with patch("builtins.__import__", _patch_missing({"psycopg2"})):
        with pytest.raises(ImportError) as exc_info:
            import agent_memory.store.postgres_backend  # noqa: F401
    assert "memwright[postgres]" in str(exc_info.value)
