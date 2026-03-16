"""Shared test fixtures -- zero-config, no Docker required."""

import tempfile

import pytest

from agent_memory import AgentMemory


# Minimal config -- no external services needed
TEST_CONFIG = {
    "default_token_budget": 2000,
    "min_results": 3,
}


@pytest.fixture
def test_config():
    """Config dict for tests."""
    return dict(TEST_CONFIG)


@pytest.fixture
def mem_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def mem(mem_dir):
    m = AgentMemory(mem_dir, config=TEST_CONFIG)
    yield m
    m.close()
