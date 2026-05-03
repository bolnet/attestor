"""BM25 / FTS retrieval lane (Phase 4.2, roadmap §B.2).

Postgres' built-in full-text search with ``ts_rank_cd`` over a generated
``content_tsv`` column gives a BM25-flavored keyword lane that runs in
parallel with the vector lane. The orchestrator fuses both via RRF
(Reciprocal Rank Fusion, k=60) to lift queries that the embedding alone
under-recalls (acronyms, exact identifiers, rare proper nouns).

Why a generated column instead of an expression index:
  - Generated tsvector with weights (A=content, B=tags, C=entity) gives
    deterministic ranking that the orchestrator can reason about.
  - GIN over a stored column is the cheapest at query time.
  - Schema migration is a single ALTER ADD COLUMN IF NOT EXISTS.

Pure document lane: no LLM, no embedder, no network. Safe to call
inside the recall hot path.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

logger = logging.getLogger("attestor.retrieval.bm25")


@dataclass(frozen=True)
class BM25Hit:
    """One BM25-ranked memory hit."""
    memory_id: str
    score: float


class BM25Lane:
    """Postgres FTS-backed keyword retrieval lane.

    Uses ``websearch_to_tsquery`` so user queries can include phrases
    ("dark mode"), boolean operators (foo OR bar), and negation (-spam)
    without the caller having to write tsquery syntax.

    Falls back gracefully when the FTS column or index isn't present
    (e.g., older v3 schema, fresh DB before the ALTER ran).
    """

    def __init__(self, conn: Any) -> None:
        self._conn = conn

    def search(
        self,
        query: str,
        *,
        limit: int = 20,
        min_rank: float = 0.0,
        active_only: bool = True,
        as_of: datetime | None = None,
        time_window: Any | None = None,
        namespace: str | None = None,
    ) -> list[BM25Hit]:
        """Return the top BM25-ranked memory ids for ``query``.

        RLS scopes the search by ``user_id`` automatically — the caller
        must already have set ``attestor.current_user_id`` on the
        connection. RLS does NOT filter by ``namespace`` (the v4 schema
        has no namespace column; sub-tenancy lives in
        ``metadata->>'_namespace'``). Callers that use namespace as a
        sub-tenancy key (LME bench: one user, many per-sample
        namespaces) MUST pass ``namespace=...`` here, otherwise BM25
        leaks rows across namespaces — same shape as the bug PR #77
        fixed for vector search. See test_tenancy_rbac_gaps.

        Phase 5.2 — bi-temporal filters (roadmap §C.2/§C.3):
          time_window → tstzrange(valid_from, valid_until) && tstzrange
          as_of       → t_created/t_expired and valid_from/valid_until
                        evaluated at that point in time

        Returns an empty list if the FTS column is missing or the query
        is empty / pure stop words.
        """
        if not query or not query.strip():
            return []
        params: dict[str, Any] = {"q": query, "lim": limit}
        where: list[str] = ["content_tsv @@ websearch_to_tsquery('english', %(q)s)"]
        if active_only:
            where.append("status = 'active'")
        if namespace is not None:
            # v4 only — v3 BM25 callers pre-date the metadata-namespace
            # convention. The FTS column itself is v4-only (the
            # ``content_tsv`` ALTER lives in the v4 schema), so this
            # branch only ever runs against v4 rows in practice.
            where.append("metadata->>'_namespace' = %(namespace)s")
            params["namespace"] = namespace
        if as_of is not None:
            where.append(
                "t_created <= %(as_of)s "
                "AND COALESCE(t_expired, 'infinity'::timestamptz) > %(as_of)s "
                "AND tstzrange(valid_from, "
                "  COALESCE(valid_until, 'infinity'::timestamptz)) @> %(as_of)s::timestamptz"
            )
            params["as_of"] = as_of
        if time_window is not None:
            params["tw_start"] = getattr(time_window, "start", None)
            params["tw_end"] = getattr(time_window, "end", None)
            where.append(
                "tstzrange(valid_from, "
                "  COALESCE(valid_until, 'infinity'::timestamptz)) "
                "&& tstzrange(%(tw_start)s::timestamptz, %(tw_end)s::timestamptz)"
            )

        sql = (
            "SELECT id, "
            "ts_rank_cd(content_tsv, websearch_to_tsquery('english', %(q)s)) AS rank "
            "FROM memories WHERE " + " AND ".join(where) +
            " ORDER BY rank DESC LIMIT %(lim)s"
        )

        try:
            with self._conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
        except Exception as e:
            # Most likely the FTS column doesn't exist yet on a v3-only
            # deployment. Log once at debug and return empty so the
            # orchestrator's other lanes still produce results.
            logger.debug("BM25 search failed (%s); returning no hits", e)
            return []

        hits: list[BM25Hit] = []
        for row in rows:
            mid, rank = row[0], float(row[1])
            if rank < min_rank:
                continue
            hits.append(BM25Hit(memory_id=str(mid), score=rank))
        return hits


def reciprocal_rank_fusion(
    *ranked_lists: list[str],
    k: int = 60,
    limit: int | None = None,
) -> list[str]:
    """Fuse multiple ranked id lists into one via RRF (Cormack '09).

    score(d) = sum over lanes of 1 / (k + rank_in_lane(d))

    k=60 is the canonical default; higher k flattens individual lanes'
    influence (good when one lane is noisy).

    Returns memory ids ordered by descending fused score.
    """
    scores: dict[str, float] = {}
    for lane in ranked_lists:
        for rank, mid in enumerate(lane):
            scores[mid] = scores.get(mid, 0.0) + 1.0 / (k + rank + 1)
    ordered = sorted(scores.keys(), key=lambda m: scores[m], reverse=True)
    if limit is not None:
        ordered = ordered[:limit]
    return ordered
