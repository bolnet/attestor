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
from typing import Any

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
        headers: dict[str, str] | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.agent_id = agent_id
        self.timeout = timeout
        self._headers = {
            "Content-Type": "application/json",
            "X-Agent-ID": agent_id,
            **(headers or {}),
        }

    def _post(self, path: str, body: dict[str, Any]) -> Any:
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
        tags: list[str] | None = None,
        category: str = "general",
        entity: str | None = None,
        event_date: str | None = None,
        confidence: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> Memory:
        """Add a memory via the HTTP API."""
        body: dict[str, Any] = {
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
        self,
        query: str,
        budget: int | None = None,
        namespace: str | None = None,
    ) -> list[RetrievalResult]:
        """Recall memories via the HTTP API.

        ``namespace`` is forwarded to the server so namespace-scoped recall
        works over HTTP (parity with the embedded ``AgentMemory.recall``).
        """
        body: dict[str, Any] = {"query": query}
        if budget:
            body["budget"] = budget
        if namespace is not None:
            body["namespace"] = namespace

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
        self,
        query: str,
        budget: int | None = None,
        namespace: str | None = None,
    ) -> str:
        """Recall and format as context string."""
        results = self.recall(query, budget=budget, namespace=namespace)
        if not results:
            return ""
        lines = []
        for r in results:
            lines.append(f"- [{r.match_source}:{r.score:.2f}] {r.memory.content}")
        return "\n".join(lines)

    def search(self, **kwargs: Any) -> list[Memory]:
        """Search memories with filters."""
        data = self._post("/search", kwargs)
        return [Memory.from_row(item) for item in data]

    def get(self, memory_id: str) -> Memory | None:
        """Get a specific memory by ID."""
        try:
            data = self._get(f"/memory/{memory_id}")
            return Memory.from_row(data)
        except (urllib.error.HTTPError, RuntimeError):
            return None

    def timeline(self, entity: str) -> list[Memory]:
        """Get chronological history for an entity."""
        data = self._post("/timeline", {"entity": entity})
        return [Memory.from_row(item) for item in data]

    def forget(self, memory_id: str) -> bool:
        """Archive a memory."""
        data = self._post("/forget", {"memory_id": memory_id})
        return data.get("forgotten", False)

    def health(self) -> dict[str, Any]:
        """Check backend health."""
        return self._get("/health")

    def stats(self) -> dict[str, Any]:
        """Get store statistics."""
        return self._get("/stats")

    def close(self) -> None:
        """No-op for HTTP client."""
        pass
