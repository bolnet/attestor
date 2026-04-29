"""Synthetic abstention dataset loader.

Plugs into the existing AbstentionBench harness
(``evals.abstention.runner.run``) by providing a 30-sample dataset of
hand-curated answerable + unanswerable cases. The fixtures live in
``evals/abstention/fixtures.json`` — 15 answerable samples mixed with
15 unanswerable across 6 categories.

This is a USER-RUN scaffold (per ``project_default_config_file``). The
loader is shipped; the user is expected to wire `mem_factory`,
`ingest`, and `answer` callables and invoke `evals.abstention.run()`.

Usage::

    from evals.abstention.runner import run as abstention_run
    from evals.abstention.synthetic_loader import SyntheticLoader

    summary = abstention_run(
        mem_factory=my_mem_factory,
        ingest=my_ingest,
        answer=my_answer_with_chain_of_note,
        loader=SyntheticLoader(),
        output_dir="docs/bench",
    )

Categories surfaced via ``AbstentionSample.category`` for per-slice F1:

  Answerable (gold = should answer):
    - personal_attribute  user states an attribute, Q asks for it
    - decision_record     a decision was made, Q asks what
    - preference          a preference change, Q asks current
    - relationship        a relationship fact, Q asks about it
    - count               a count, Q asks for it
    - professional        a job-related fact, Q asks for it

  Unanswerable (gold = should abstain):
    - unknown_topic       Q is on a topic the haystack doesn't cover
    - false_premise       Q presumes a fact the haystack contradicts
    - future_event        Q asks about a not-yet-happened event
    - underspecified      Q asks for specifics not in context
    - subjective_opinion  Q asks for a value judgment
    - absent_relationship Q asks about a relationship neither mentioned
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from evals.abstention.types import AbstentionSample


_DEFAULT_FIXTURES = (
    Path(__file__).resolve().parent / "fixtures.json"
)


class SyntheticLoader:
    """DatasetLoader for the existing AbstentionBench harness.

    Reads ``evals/abstention/fixtures.json`` and returns one
    ``AbstentionSample`` per case. Stable order so reports are
    diffable across runs.
    """

    def __init__(self, fixtures_path: Optional[Path | str] = None):
        self.fixtures_path = Path(fixtures_path or _DEFAULT_FIXTURES)

    def __call__(self) -> List[AbstentionSample]:
        return load_synthetic_samples(self.fixtures_path)


def load_synthetic_samples(
    fixtures_path: Path | str = _DEFAULT_FIXTURES,
) -> List[AbstentionSample]:
    """Read fixtures.json and produce AbstentionSamples.

    Strict on shape — missing required fields raise ValueError so a
    typo can't silently produce an empty bench.
    """
    p = Path(fixtures_path)
    raw = json.loads(p.read_text())
    if not isinstance(raw, dict) or not isinstance(raw.get("samples"), list):
        raise ValueError(
            f"abstention fixtures: expected {{ 'samples': [...] }} in {p}; "
            f"got top-level keys {list(raw)!r}"
        )

    out: List[AbstentionSample] = []
    for case in raw["samples"]:
        for required in ("sample_id", "category", "answerable", "context", "query"):
            if required not in case:
                raise ValueError(
                    f"abstention fixture {case.get('sample_id', '?')!r} "
                    f"missing required field {required!r}"
                )
        out.append(AbstentionSample(
            sample_id=str(case["sample_id"]),
            context=str(case["context"]),
            query=str(case["query"]),
            answerable=bool(case["answerable"]),
            expected_answer=case.get("expected_answer"),
            category=str(case["category"]),
            metadata=dict(case.get("metadata") or {}),
        ))

    if not out:
        raise ValueError(
            f"abstention fixtures: {p} contains no samples — "
            f"expected ≥ 1 case",
        )
    return out
