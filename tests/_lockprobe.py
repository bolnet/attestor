"""Test helper: is a (re-entrant) lock currently owned by the calling thread?

``threading.RLock`` exposes no public ownership query, so the probe asks a
second thread to take the lock without blocking — it fails exactly when
the caller (or anyone else) holds it. Used by the connection-lock tests to
assert SQL runs *inside* the lock rather than merely near it.
"""

from __future__ import annotations

import threading
from typing import Any


def held_by_caller(lock: Any) -> bool:
    outcome: list[bool] = []

    def probe() -> None:
        acquired = lock.acquire(blocking=False)
        if acquired:
            lock.release()
        outcome.append(not acquired)

    worker = threading.Thread(target=probe)
    worker.start()
    worker.join()
    return outcome[0]
