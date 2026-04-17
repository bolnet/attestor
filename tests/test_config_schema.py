import pytest
from pydantic import ValidationError

from agent_memory.config.schema import MemwrightSettings


def test_defaults():
    s = MemwrightSettings()
    assert s.backends == ["sqlite", "chroma", "networkx"]
    assert s.default_token_budget == 16000
    assert s.fusion_mode == "rrf"


def test_validates_fusion_mode():
    with pytest.raises(ValidationError):
        MemwrightSettings(fusion_mode="invalid_mode")


def test_validates_token_budget_positive():
    with pytest.raises(ValidationError):
        MemwrightSettings(default_token_budget=-1)


def test_arango_settings_nested():
    s = MemwrightSettings(
        backends=["arangodb"],
        arangodb={"mode": "local", "port": 8529},
    )
    assert s.arangodb.mode == "local"
    assert s.arangodb.port == 8529


def test_validates_backends_non_empty():
    with pytest.raises(ValidationError):
        MemwrightSettings(backends=[])
