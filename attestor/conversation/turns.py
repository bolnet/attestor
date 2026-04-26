"""ConversationTurn — speaker-tagged verbatim message dataclass.

Roadmap §A.1: a turn is one message. A *round* is a (user_turn,
assistant_turn) pair. This module models the single turn; the round
is composed in ``ConversationIngest.ingest_round``.

Why frozen: turns are append-only audit records. Once captured, they
must never be paraphrased, edited, or reordered. Mutation here would
break the bi-temporal guarantee that ``episodes`` is the immutable
source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class ConversationTurn:
    """One message in a conversation. Verbatim — never paraphrased.

    Attributes:
        thread_id: Stable identifier for the conversation thread. Multiple
            sessions may share a thread_id (e.g., resumed chat); episodes
            join on this for thread-level history.
        speaker: One of ``"user" | "assistant" | "<agent_id>"``. Used by
            extraction prompts to lock to one speaker (the speaker-lock
            IMPORTANT line in roadmap §A.2).
        role: OpenAI/Anthropic-style role: ``"user" | "assistant" |
            "system" | "tool"``. Distinct from ``speaker`` so we can
            track multi-agent provenance while still sending standard
            chat-completion roles to the LLM.
        content: The verbatim text of the message.
        ts: Event time (when the turn occurred). Used for temporal
            reasoning + ``valid_from`` of any extracted facts.
        parent_turn_id: Optional pointer to the previous turn id, for
            tool-call chains and forks.
        tool_calls: Optional list of tool invocations attached to this
            turn (assistant-side typically).
        metadata: Arbitrary JSON-serializable extras.
    """

    thread_id: str
    speaker: str
    role: str
    content: str
    ts: datetime = field(default_factory=_now_utc)
    parent_turn_id: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.thread_id:
            raise ValueError("ConversationTurn.thread_id is required")
        if not self.speaker:
            raise ValueError("ConversationTurn.speaker is required")
        if self.role not in {"user", "assistant", "system", "tool"}:
            raise ValueError(
                f"ConversationTurn.role must be one of "
                f"user/assistant/system/tool; got {self.role!r}"
            )
        if not self.content:
            raise ValueError("ConversationTurn.content cannot be empty")

    @property
    def is_user(self) -> bool:
        return self.role == "user"

    @property
    def is_assistant(self) -> bool:
        return self.role == "assistant"
