"""Braintrust eval: attestor recall pipeline on LOCOMO, judged by Claude Opus 4.7.

Mirrors the server-side `factuality-b2d8` scorer (Opus 4.7 + CoT, 1.0/0.5/0.0)
as a local Python scorer for reliable iteration.

Run:
    set -a && source .env && set +a
    .venv/bin/python evals/braintrust_locomo.py --max-conversations 1 --max-questions 3

Or via braintrust CLI:
    .venv/bin/braintrust eval evals/braintrust_locomo.py
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname).1s %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("eval")
logging.getLogger("attestor").setLevel(logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)

from anthropic import Anthropic
from braintrust import Eval, init_dataset

from urllib.parse import urlparse

from attestor import AgentMemory
from attestor.locomo import (
    CATEGORY_NAMES,
    answer_question,
    load_locomo,
)
from attestor.extraction.extractor import extract_from_session

PROJECT = "attestor-mab"
DATASET = "locomo-v1"
DEFAULT_DATA_PATH = str(Path.home() / ".cache" / "attestor" / "locomo10.json")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "claude-opus-4-7")
# All generation via Opus 4.7 (OpenRouter). Answer + extraction + reflection.
ANSWER_MODEL = os.environ.get("ANSWER_MODEL", "anthropic/claude-opus-4.7")
EXTRACTION_MODEL = os.environ.get("EXTRACTION_MODEL", "anthropic/claude-opus-4.7")
RECALL_BUDGET = 6000

FACTUALITY_PROMPT = """You are grading an answer produced by an AI memory system.
Compare the predicted answer to the reference answer.

Question: {question}

Reference answer:
{reference}

Predicted answer:
{prediction}

Score rubric:
- 1.0 = CORRECT: fully entailed by the reference; acceptable paraphrases count
- 0.5 = PARTIAL: touches the right topic but is incomplete or vague
- 0.0 = INCORRECT: contradictory, irrelevant, or "I don't know"

Think step by step, then output a single JSON object on the last line:
{{"reasoning": "<one sentence>", "score": <0.0|0.5|1.0>}}"""


def _flatten_qa(
    data_path: str,
    max_conversations: int | None,
    max_questions: int | None,
) -> list[dict[str, Any]]:
    """Flatten LOCOMO QA pairs into per-question rows."""
    convs = load_locomo(data_path)
    if max_conversations:
        convs = convs[:max_conversations]

    rows: list[dict[str, Any]] = []
    for conv in convs:
        qa_pairs = conv["qa"]
        if max_questions:
            qa_pairs = qa_pairs[:max_questions]
        for qa in qa_pairs:
            rows.append(
                {
                    "input": {
                        "question": qa["question"],
                        "sample_id": conv["sample_id"],
                        "speaker_a": conv["speaker_a"],
                        "speaker_b": conv["speaker_b"],
                    },
                    "expected": str(qa["answer"]),
                    "metadata": {
                        "sample_id": conv["sample_id"],
                        "category": qa.get("category"),
                        "category_name": CATEGORY_NAMES.get(
                            qa.get("category"), "unknown"
                        ),
                        "evidence": qa.get("evidence", []),
                    },
                }
            )
    return rows


def _postgres_config() -> dict[str, Any]:
    """Build Layer 0 AgentMemory config: Postgres (doc+vector) + Neo4j (graph).

    Neo4j is wired when NEO4J_URI is set; otherwise graph role is disabled.
    """
    url = os.environ.get("NEON_DATABASE_URL") or os.environ.get("PG_CONNECTION_STRING")
    if not url:
        raise SystemExit("NEON_DATABASE_URL or PG_CONNECTION_STRING required")
    p = urlparse(url)
    host = p.hostname or "localhost"
    port = p.port or 5432
    db = (p.path or "/").lstrip("/") or "attestor"
    cfg: dict[str, Any] = {
        "backends": ["postgres"],
        "postgres": {
            "url": f"postgresql://{host}:{port}",
            "database": db,
            "auth": {"username": p.username or "", "password": p.password or ""},
            "sslmode": os.environ.get("PG_SSLMODE", "require"),
        },
    }
    neo4j_uri = os.environ.get("NEO4J_URI")
    if neo4j_uri:
        cfg["backends"].append("neo4j")
        cfg["neo4j"] = {
            "url": neo4j_uri,
            "database": os.environ.get("NEO4J_DATABASE", "neo4j"),
            "auth": {
                "username": os.environ.get("NEO4J_USERNAME", "neo4j"),
                "password": os.environ.get("NEO4J_PASSWORD", ""),
            },
        }
    return cfg


def _dual_path_ingest(
    mem: AgentMemory,
    conv: dict[str, Any],
    *,
    api_key: str | None,
    verbose: bool = False,
) -> dict[str, int]:
    """Dual-path ingest. Path A = raw turn chunks; Path B = LLM extraction.

    Per-turn / per-fact / per-triple trace logging.
    """
    stats = {"chunks": 0, "facts": 0, "triples": 0, "graph_failed": 0}
    speaker_a = conv["speaker_a"]
    speaker_b = conv["speaker_b"]
    sid = conv["sample_id"]
    log.info("[%s] ingest start: %d sessions", sid, len(conv["sessions"]))

    for s_idx, session in enumerate(conv["sessions"]):
        session_id = f"session_{session['session_id']}"
        date_time = session.get("date_time", "")
        turns = session["turns"]
        t_session = time.perf_counter()

        # ── Path A: raw chunks ──
        t_a = time.perf_counter()
        a_count = 0
        for t_idx, turn in enumerate(turns):
            text = turn.get("text", "")
            if not text:
                continue
            sp = turn.get("speaker", "?")
            display = speaker_a if sp == "A" else speaker_b if sp == "B" else sp
            t0 = time.perf_counter()
            result = mem.add(
                content=f"{display}: {text}",
                tags=["chunk", display.lower(), session_id],
                category="chunk",
                entity=display,
                event_date=date_time,
                metadata={
                    "path": "A",
                    "namespace": "document_chunks",
                    "session": session_id,
                    "dia_id": turn.get("dia_id", ""),
                },
            )
            ms = (time.perf_counter() - t0) * 1000
            stats["chunks"] += 1
            a_count += 1
            if verbose:
                preview = (text[:60] + "…") if len(text) > 60 else text
                log.debug(
                    "[%s/%s t%d] A+chunk id=%s ms=%.1f: %s: %s",
                    sid, session_id, t_idx,
                    getattr(result, "id", "?")[:8], ms, display, preview,
                )
        log.info(
            "[%s/%s] Path A done: %d chunks in %.2fs",
            sid, session_id, a_count, time.perf_counter() - t_a,
        )

        # ── Path B: LLM extraction ──
        t_b = time.perf_counter()
        log.info(
            "[%s/%s] Path B extraction calling %s (%d turns)...",
            sid, session_id, EXTRACTION_MODEL, len(turns),
        )
        memories, triples = extract_from_session(
            turns=turns,
            speaker_a=speaker_a,
            speaker_b=speaker_b,
            session_date=date_time,
            model=EXTRACTION_MODEL,
            api_key=api_key,
        )
        log.info(
            "[%s/%s] Path B LLM returned: %d facts, %d triples in %.2fs",
            sid, session_id, len(memories), len(triples),
            time.perf_counter() - t_b,
        )

        for f_idx, m in enumerate(memories):
            t0 = time.perf_counter()
            result = mem.add(
                content=m.content,
                tags=list(m.tags) + ["entity"],
                category="entity",
                entity=m.entity,
                event_date=m.event_date or date_time,
                confidence=m.confidence,
                metadata={
                    "path": "B",
                    "namespace": "entities",
                    "session": session_id,
                },
            )
            ms = (time.perf_counter() - t0) * 1000
            stats["facts"] += 1
            if verbose:
                preview = (m.content[:80] + "…") if len(m.content) > 80 else m.content
                log.debug(
                    "[%s/%s f%d] B+fact id=%s ent=%s ms=%.1f: %s",
                    sid, session_id, f_idx,
                    getattr(result, "id", "?")[:8],
                    m.entity, ms, preview,
                )

        if triples and mem._graph:
            for tr_idx, t in enumerate(triples):
                subj, pred, obj = t["subject"], t["predicate"], t["object"]
                try:
                    mem._graph.add_entity(subj, "person")
                    mem._graph.add_entity(obj, "entity")
                    mem._graph.add_relation(
                        subj, obj, pred,
                        metadata={
                            "event_date": t.get("event_date") or date_time,
                            "session": session_id,
                        },
                    )
                    stats["triples"] += 1
                    if verbose:
                        log.debug(
                            "[%s/%s tr%d] B+triple: (%s) -[%s {date:%s}]-> (%s)",
                            sid, session_id, tr_idx, subj, pred,
                            t.get("event_date") or date_time, obj,
                        )
                except NotImplementedError:
                    stats["graph_failed"] += 1
                    if stats["graph_failed"] == 1:
                        log.warning(
                            "[%s] graph role unavailable (AGE not loaded); "
                            "skipping triples",
                            sid,
                        )
                except Exception as e:
                    log.warning(
                        "[%s/%s tr%d] graph add failed: %s",
                        sid, session_id, tr_idx, e,
                    )
                    stats["graph_failed"] += 1

        log.info(
            "[%s/%s] session done: A=%d B-facts=%d B-triples=%d "
            "(graph_failed=%d) session_time=%.2fs",
            sid, session_id, a_count, len(memories),
            stats["triples"] - (stats["triples"] - len(triples) if triples else 0),
            stats["graph_failed"], time.perf_counter() - t_session,
        )

    log.info(
        "[%s] ingest DONE: chunks=%d facts=%d triples=%d graph_failed=%d",
        sid, stats["chunks"], stats["facts"], stats["triples"],
        stats["graph_failed"],
    )
    return stats


def _precompute_predictions(
    convs: list[dict[str, Any]],
    max_questions: int | None,
    api_key: str | None,
    enable_reflection: bool = False,
    debug: bool = False,
) -> dict[tuple[str, str], str]:
    """Dual-path ingest into Postgres, answer each question."""
    predictions: dict[tuple[str, str], str] = {}
    pg_config = _postgres_config()

    for conv in convs:
        with tempfile.TemporaryDirectory() as tmpdir:
            mem = AgentMemory(tmpdir, config=pg_config)
            if mem._retrieval:
                mem._retrieval.enable_temporal_boost = False
            if mem._temporal:
                mem._temporal.check_contradictions = lambda m: []

            print(f"  [{conv['sample_id']}] dual-path ingest starting...")
            stats = _dual_path_ingest(mem, conv, api_key=api_key, verbose=debug)
            print(
                f"  [{conv['sample_id']}] chunks={stats['chunks']} "
                f"facts={stats['facts']} triples={stats['triples']} "
                f"graph_failed={stats['graph_failed']}"
            )

            qa_pairs = conv["qa"]
            if max_questions:
                qa_pairs = qa_pairs[:max_questions]
            for qa in qa_pairs:
                pred = answer_question(
                    mem,
                    qa["question"],
                    budget=RECALL_BUDGET,
                    model=ANSWER_MODEL,
                    api_key=api_key,
                    speaker_a=conv["speaker_a"],
                    speaker_b=conv["speaker_b"],
                    enable_reflection=enable_reflection,
                )
                predictions[(conv["sample_id"], qa["question"])] = pred
                if debug:
                    print(f"  Q: {qa['question']}")
                    print(f"  expected : {qa['answer']}")
                    print(f"  predicted: {pred}")
                    print()
            mem.close()
    return predictions


def _judge(
    client: Anthropic,
    question: str,
    reference: str,
    prediction: str,
) -> dict[str, Any]:
    msg = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=512,
        messages=[
            {
                "role": "user",
                "content": FACTUALITY_PROMPT.format(
                    question=question,
                    reference=reference,
                    prediction=prediction,
                ),
            }
        ],
    )
    text = msg.content[0].text.strip()
    last_line = text.rsplit("\n", 1)[-1].strip()
    try:
        parsed = json.loads(last_line)
        score = float(parsed.get("score", 0.0))
        score = max(0.0, min(1.0, score))
        return {
            "score": score,
            "metadata": {"judge_reasoning": parsed.get("reasoning", "")[:500]},
        }
    except Exception as exc:
        return {
            "score": 0.0,
            "metadata": {"judge_error": str(exc), "raw": text[-300:]},
        }


def upload_dataset(
    data_path: str,
    max_conversations: int | None,
    max_questions: int | None,
) -> None:
    """One-shot upload LOCOMO rows to Braintrust dataset locomo-v1."""
    rows = _flatten_qa(data_path, max_conversations, max_questions)
    ds = init_dataset(project=PROJECT, name=DATASET)
    for row in rows:
        ds.insert(
            input=row["input"],
            expected=row["expected"],
            metadata=row["metadata"],
        )
    summary = ds.summarize()
    print(f"Uploaded {len(rows)} rows to {PROJECT}/{DATASET}")
    print(summary)


def run(
    data_path: str,
    max_conversations: int | None,
    max_questions: int | None,
    experiment_suffix: str,
    resolve_pronouns: bool = False,
    enable_reflection: bool = False,
    debug: bool = False,
) -> None:
    if not os.environ.get("BRAINTRUST_API_KEY"):
        raise SystemExit(
            "BRAINTRUST_API_KEY not set. Run: set -a && source .env && set +a"
        )
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY not set.")
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit("OPENROUTER_API_KEY not set.")

    convs = load_locomo(data_path)
    if max_conversations:
        convs = convs[:max_conversations]

    api_key = os.environ.get("OPENROUTER_API_KEY")

    print(
        f"Precomputing predictions for {len(convs)} conversation(s), "
        f"{max_questions or 'all'} question(s) each "
        f"[resolve_pronouns={resolve_pronouns}, enable_reflection={enable_reflection}]..."
    )
    # `resolve_pronouns` is not wired into the dual-path ingest yet; retained
    # on the CLI surface for parity with the legacy single-path evaluator.
    del resolve_pronouns
    predictions = _precompute_predictions(
        convs,
        max_questions,
        api_key,
        enable_reflection=enable_reflection,
        debug=debug,
    )
    print(f"Got {len(predictions)} predictions.")

    rows = _flatten_qa(data_path, max_conversations, max_questions)
    anthropic = Anthropic()

    def factuality_scorer(input, output, expected, metadata):
        res = _judge(anthropic, input["question"], expected, output)
        return {
            "name": "factuality_opus47",
            "score": res["score"],
            "metadata": res["metadata"],
        }

    Eval(
        PROJECT,
        data=lambda: [
            {"input": r["input"], "expected": r["expected"], "metadata": r["metadata"]}
            for r in rows
        ],
        task=lambda row_input: predictions[
            (row_input["sample_id"], row_input["question"])
        ],
        scores=[factuality_scorer],
        experiment_name=f"locomo-{experiment_suffix}",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DEFAULT_DATA_PATH)
    parser.add_argument("--max-conversations", type=int, default=1)
    parser.add_argument("--max-questions", type=int, default=3)
    parser.add_argument(
        "--upload-dataset",
        action="store_true",
        help="Upload rows to Braintrust dataset locomo-v1 and exit.",
    )
    parser.add_argument(
        "--suffix",
        default="smoke",
        help="Experiment name suffix (e.g., smoke, full).",
    )
    parser.add_argument("--resolve-pronouns", action="store_true")
    parser.add_argument("--enable-reflection", action="store_true")
    parser.add_argument("--debug", action="store_true", help="Print Q/expected/predicted locally.")
    args = parser.parse_args()

    if args.upload_dataset:
        upload_dataset(args.data, args.max_conversations, args.max_questions)
    else:
        run(
            args.data,
            args.max_conversations,
            args.max_questions,
            args.suffix,
            resolve_pronouns=args.resolve_pronouns,
            enable_reflection=args.enable_reflection,
            debug=args.debug,
        )
