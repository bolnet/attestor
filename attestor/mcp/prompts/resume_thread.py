"""resume_thread — chronological recall of a thread's recent activity.

The "I'm back, what was happening here?" primitive. Issues a
recall_context call scoped to the thread, restricted to the last N
days, sorted chronologically (oldest → newest).
"""

from __future__ import annotations

RESUME_THREAD_PROMPT = """\
You are resuming work on thread {thread_id}. You have not been active
on it for some time. Read the recall results below in chronological
order (oldest → newest) and produce:

1. STATE — one paragraph describing where the thread currently stands.
2. LAST ACTION — what was the most recent decision or commitment?
3. OPEN ITEMS — what was waiting on action when the thread paused?
4. STALE — anything in the recall that is now obsolete (cite [mem_<id>]).

# IMPORTANT: This is a READING task, not a deciding task. Do not take
# new actions. After producing the four sections, ask the user to
# confirm the next step.

Recall window: last {window_days} days
Memories (chronological):
{memories_chronological}
"""


def format_resume_thread_prompt(
    *,
    thread_id: str,
    memories_chronological: str,
    window_days: int = 30,
) -> str:
    return RESUME_THREAD_PROMPT.format(
        thread_id=thread_id,
        memories_chronological=memories_chronological,
        window_days=window_days,
    )
