"""MemoryClient -- thin HTTP client for distributed multi-agent memory access.

Mirrors the AgentMemory interface but delegates to the Starlette ASGI API
over HTTP. Used by AgentContext when memory_url is set instead of a local path.

Usage:
    client = MemoryClient("https://memory.internal", agent_id="planner-01")
    client.add("User prefers Python", tags=["preference"])
    results = client.recall("language preferences")
"""

from __future__ import annotations

import json
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional

from attestor.models import Memory, RetrievalResult


class MemoryClient:
    """HTTP client for the Attestor ASGI API.

    Zero-dependency -- uses only stdlib urllib. Drop-in replacement
    for AgentMemory in read/write operations.
    """

    def __init__(
        self,
        base_url: str,
        agent_id: str = "anonymous",
        timeout: float = 10.0,
        headers: Optional[Dict[str, str]] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.agent_id = agent_id
        self.timeout = timeout
        self._headers = {
            "Content-Type": "application/json",
            "X-Agent-ID": agent_id,
            **(headers or {}),
        }

    def _post(self, path: str, body: Dict[str, Any]) -> Any:
        """POST JSON to the API and return parsed response data."""
        url = f"{self.base_url}{path}"
        data = json.dumps(body).encode()
        req = urllib.request.Request(
            url, data=data, headers=self._headers, method="POST"
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            result = json.loads(resp.read())
        if not result.get("ok"):
            raise RuntimeError(result.get("error", "Unknown API error"))
        return result["data"]

    def _get(self, path: str) -> Any:
        """GET from the API and return parsed response data."""
        url = f"{self.base_url}{path}"
        req = urllib.request.Request(url, headers=self._headers, method="GET")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            result = json.loads(resp.read())
        if not result.get("ok"):
            raise RuntimeError(result.get("error", "Unknown API error"))
        return result["data"]

    # -- Write --

    def add(
        self,
        content: str,
        tags: Optional[List[str]] = None,
        category: str = "general",
        entity: Optional[str] = None,
        event_date: Optional[str] = None,
        confidence: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Memory:
        """Add a memory via the HTTP API."""
        body: Dict[str, Any] = {
            "content": content,
            "tags": tags or [],
            "category": category,
            "confidence": confidence,
            "metadata": {
                **(metadata or {}),
                "_agent_id": self.agent_id,
            },
        }
        if entity:
            body["entity"] = entity
        if event_date:
            body["event_date"] = event_date

        data = self._post("/add", body)
        return Memory.from_row(data)

    # -- Read --

    def recall(
        self, query: str, budget: Optional[int] = None
    ) -> List[RetrievalResult]:
        """Recall memories via the HTTP API."""
        body: Dict[str, Any] = {"query": query}
        if budget:
            body["budget"] = budget

        data = self._post("/recall", body)
        results = []
        for item in data:
            mem = Memory.from_row(item["memory"])
            results.append(RetrievalResult(
                memory=mem,
                score=item["score"],
                match_source=item["source"],
            ))
        return results

    def recall_as_context(
        self, query: str, budget: Optional[int] = None
    ) -> str:
        """Recall and format as context string."""
        results = self.recall(query, budget=budget)
        if not results:
            return ""
        lines = []
        for r in results:
            lines.append(f"- [{r.match_source}:{r.score:.2f}] {r.memory.content}")
        return "\n".join(lines)

    def search(self, **kwargs: Any) -> List[Memory]:
        """Search memories with filters."""
        data = self._post("/search", kwargs)
        return [Memory.from_row(item) for item in data]

    def get(self, memory_id: str) -> Optional[Memory]:
        """Get a specific memory by ID."""
        try:
            data = self._get(f"/memory/{memory_id}")
            return Memory.from_row(data)
        except (urllib.error.HTTPError, RuntimeError):
            return None

    def timeline(self, entity: str) -> List[Memory]:
        """Get chronological history for an entity."""
        data = self._post("/timeline", {"entity": entity})
        return [Memory.from_row(item) for item in data]

    def forget(self, memory_id: str) -> bool:
        """Archive a memory."""
        data = self._post("/forget", {"memory_id": memory_id})
        return data.get("forgotten", False)

    def health(self) -> Dict[str, Any]:
        """Check backend health."""
        return self._get("/health")

    def stats(self) -> Dict[str, Any]:
        """Get store statistics."""
        return self._get("/stats")

    def close(self) -> None:
        """No-op for HTTP client."""
        pass
