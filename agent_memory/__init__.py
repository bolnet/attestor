"""AgentMemory — Embedded memory for AI agents."""

from agent_memory._version import get_version
from agent_memory.core import AgentMemory
from agent_memory.models import Memory, RetrievalResult

__all__ = ["AgentMemory", "Memory", "RetrievalResult"]
__version__ = get_version()
