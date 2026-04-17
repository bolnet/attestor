"""Tests for the require_extra helper."""

import pytest

from agent_memory.store._extras import require_extra


def test_require_extra_passes_when_present():
    """Test that require_extra successfully imports an available module."""
    # Import json which is always available in the stdlib
    module = require_extra("json", extra="test", package="memwright")
    assert module is not None


def test_require_extra_raises_helpful_error():
    """Test that require_extra raises an ImportError with actionable message."""
    with pytest.raises(ImportError) as exc_info:
        require_extra(
            "definitely_not_a_module_xyz",
            extra="arangodb",
            package="memwright"
        )

    error_message = str(exc_info.value)
    assert "memwright[arangodb]" in error_message
    assert "pip install" in error_message
