"""Live Temporal server smoke — env-gated on ATTESTOR_TEMPORAL_ADDRESS.

Run against a dev server::

    temporal server start-dev
    ATTESTOR_TEMPORAL_ADDRESS=localhost:7233 ATTESTOR_TEMPORAL_NAMESPACE=default \
        .venv/bin/pytest tests/test_durable_live.py -q

Exercises connect → worker → workflow → describe with a stub activity.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone

import pytest

temporalio = pytest.importorskip("temporalio")
from attestor.durable.models import ConsolidationOutcome, EpisodeRef  # noqa: E402

ADDRESS = os.environ.get("ATTESTOR_TEMPORAL_ADDRESS")
NAMESPACE = os.environ.get("ATTESTOR_TEMPORAL_NAMESPACE", "default")

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not ADDRESS, reason="ATTESTOR_TEMPORAL_ADDRESS not set"),
]


async def test_live_consolidate_workflow_round_trip():
    pytest.importorskip("temporalio")
    from temporalio import activity

    from attestor.config import DurableCfg
    from attestor.durable import client as durable_client
    from attestor.durable.client import consolidation_result, start_consolidation
    from attestor.durable.models import CONSOLIDATE_EPISODE_ACTIVITY, ConsolidateRequest
    from attestor.durable.status import describe_workflow, probe
    from attestor.durable.worker import build_worker

    cfg = DurableCfg(
        enabled=True, address=ADDRESS, namespace=NAMESPACE,
        task_queue=f"attestor-live-{uuid.uuid4().hex[:8]}",
    )
    health = await probe(cfg)
    assert health.reachable, health.error

    @activity.defn(name=CONSOLIDATE_EPISODE_ACTIVITY)
    async def stub(ref: EpisodeRef) -> ConsolidationOutcome:
        return ConsolidationOutcome(episode_id=ref.episode_id, ok=True)

    ts = datetime.now(tz=timezone.utc)
    ref = EpisodeRef(
        episode_id=f"live-{uuid.uuid4().hex[:8]}", user_id="u", session_id="s",
        thread_id="t", user_ts=ts, assistant_ts=ts,
    )
    client = await durable_client.connect(cfg)
    async with build_worker(client, cfg, activities=[stub]):
        wid = await start_consolidation(client, cfg, ConsolidateRequest(episode=ref))
        out = await consolidation_result(client, wid)
    assert out.ok
    status = await describe_workflow(cfg, wid)
    assert status.status == "COMPLETED"
