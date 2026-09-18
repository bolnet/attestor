"""Postgres document store — ``list_memory_ids`` keyset paging (RebuildDerived source).

Live-gated on POSTGRES_URL via the shared ``mem`` fixture.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

pytestmark = pytest.mark.live


def _ids(mem, **kw) -> tuple[list[str], str | None]:
    return mem._store.list_memory_ids(**{
        "since": None, "namespace": None, "user_id": None, "after_id": None, "limit": 100, **kw,
    })


def test_list_memory_ids_filters_by_owner(mem):
    """RebuildDerived pages per tenant: another user's id sees nothing."""
    import uuid

    mine = mem.add("scoped fact", namespace="lme_u").id
    owner = mem._store.get(mine).user_id
    if owner is None:
        pytest.skip("owner filter needs the v4 schema (user_id column); fixture is v3")
    assert mine in _ids(mem, user_id=owner)[0]
    assert _ids(mem, user_id=str(uuid.uuid4())) == ([], None)


def test_list_memory_ids_pages_in_id_order_and_filters(mem):
    created = [mem.add(f"fact {i}", namespace="lme_x").id for i in range(5)]
    other = mem.add("elsewhere", namespace="lme_y").id
    all_ids, nxt = _ids(mem)
    assert set(all_ids) == set(created) | {other}
    assert all_ids == sorted(all_ids)
    assert nxt is None

    page1, cursor = _ids(mem, limit=3)
    assert len(page1) == 3
    assert cursor == page1[-1]
    page2, cursor2 = _ids(mem, after_id=cursor, limit=3)
    assert page1 + page2 == all_ids
    # Keyset paging emits a cursor whenever a page is full, even when it is
    # the last one; the following fetch is the empty, cursor-less page.
    assert cursor2 == page2[-1]
    assert _ids(mem, after_id=cursor2, limit=3) == ([], None)

    scoped, _ = _ids(mem, namespace="lme_x")
    assert set(scoped) == set(created)

    future = datetime.now(timezone.utc) + timedelta(days=1)
    assert _ids(mem, since=future) == ([], None)
    past = datetime.now(timezone.utc) - timedelta(days=1)
    assert set(_ids(mem, since=past)[0]) == set(all_ids)


def test_list_memory_ids_skips_archived(mem):
    keep = mem.add("keep me", namespace="lme_z").id
    gone = mem.add("forget me", namespace="lme_z").id
    assert mem.forget(gone)
    ids, _ = _ids(mem, namespace="lme_z")
    assert keep in ids
    assert gone not in ids
