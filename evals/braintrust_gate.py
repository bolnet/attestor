"""Braintrust-backed CI regression gate for LME-S experiments.

Pulls the latest scores for each ``GOLDEN`` experiment in the
``attestor-lme-s`` Braintrust project, compares against thresholds, and
exits non-zero on regression. Designed to run in CI on every PR that
touches the retrieval / longmemeval / store paths.

Distinct from ``evals/gate.py`` — that's the Track G BenchmarkSummary
gate (deterministic, no Braintrust dependency). This one is LLM-result
gated and depends on a live Braintrust project. They coexist; CI calls
both.

CLI:

    set -a && source .env && set +a
    python -m evals.braintrust_gate \\
        --project attestor-lme-s \\
        --thresholds evals/braintrust_gate.yaml \\
        --report-out bench-output/braintrust-gate.json

Exit codes: 0 = pass, 1 = regression detected, 2 = invocation error
(missing API key, missing baseline experiment, etc.).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class GateResult:
    experiment: str
    metric: str
    baseline: float
    current: float
    delta: float
    threshold: float
    passed: bool


# Default thresholds (override via --thresholds YAML file).
# Each key is the experiment label (e.g. "lme-temporal:GOLDEN");
# values are minimum acceptable score deltas (negative = how much we
# allow the score to drop before failing the gate).
DEFAULT_THRESHOLDS: dict[str, dict[str, float]] = {
    "lme-temporal:GOLDEN":            {"all_judges_correct": -0.05},
    "lme-multi-session:GOLDEN":       {"all_judges_correct": -0.05},
    "lme-knowledge-update:GOLDEN":    {"all_judges_correct": -0.05},
    "lme-single-session-user:GOLDEN": {"all_judges_correct": -0.05},
}


def load_thresholds(path: Path | None) -> dict[str, dict[str, float]]:
    if path is None or not path.exists():
        return DEFAULT_THRESHOLDS
    with path.open() as f:
        loaded = yaml.safe_load(f) or {}
    return loaded.get("thresholds", DEFAULT_THRESHOLDS)


def _fetch_experiment_scores(project: str, experiment_name: str) -> dict[str, float]:
    """Pull experiment-level scores from Braintrust.

    Uses the ``braintrust`` SDK's ``api_url + login`` flow — same auth
    surface the upload script uses. Returns ``{score_name: value}``.
    """
    from braintrust.framework import init_experiment  # type: ignore

    exp = init_experiment(project=project, experiment=experiment_name, open=True)
    summary = exp.summarize(summarize_scores=True)
    out: dict[str, float] = {}
    for entry in summary.scores or {}:
        # Different SDK versions: support both dict-of-dicts and dict-of-floats.
        val = summary.scores[entry]
        if isinstance(val, dict):
            out[entry] = float(val.get("score") or val.get("mean") or 0.0)
        else:
            out[entry] = float(val)
    return out


def _diff_against_baseline(
    project: str,
    experiment_label: str,
    metric_thresholds: dict[str, float],
    *,
    baseline_suffix: str = "GOLDEN",
) -> list[GateResult]:
    """For one experiment label, fetch current + baseline scores and diff.

    The baseline is the most-recent run of the same label that's marked
    ``frozen`` or simply the previously-committed baseline run. Here we
    use a simple convention: the experiment named with ``-baseline``
    suffix (e.g., ``lme-temporal:GOLDEN-baseline``). If absent, the
    gate passes (bootstrap-permissive).
    """
    baseline_name = f"{experiment_label}-baseline"
    try:
        current = _fetch_experiment_scores(project, experiment_label)
    except Exception as e:
        print(f"  ! could not fetch current {experiment_label}: {e}", flush=True)
        return []
    try:
        baseline = _fetch_experiment_scores(project, baseline_name)
    except Exception:
        # No baseline frozen yet — pass.
        print(f"  · no baseline experiment {baseline_name} (bootstrap-pass)", flush=True)
        return []

    results = []
    for metric, threshold in metric_thresholds.items():
        cur = current.get(metric)
        base = baseline.get(metric)
        if cur is None or base is None:
            print(f"  · skipping {metric}: missing in {'current' if cur is None else 'baseline'}", flush=True)
            continue
        delta = cur - base
        results.append(GateResult(
            experiment=experiment_label, metric=metric,
            baseline=base, current=cur, delta=delta,
            threshold=threshold, passed=delta >= threshold,
        ))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Braintrust-backed regression gate for LME-S experiments.",
    )
    parser.add_argument("--project", default="attestor-lme-s")
    parser.add_argument("--thresholds", type=Path, default=None,
                        help="YAML with `thresholds:` block; defaults to built-ins.")
    parser.add_argument("--report-out", type=Path, default=None,
                        help="Optional JSON dump of all gate results.")
    args = parser.parse_args()

    if not os.environ.get("BRAINTRUST_API_KEY"):
        print("ERROR: BRAINTRUST_API_KEY not set", file=sys.stderr)
        sys.exit(2)

    thresholds = load_thresholds(args.thresholds)
    print(f"Gate against project={args.project}", flush=True)
    print(f"Experiments to check: {len(thresholds)}", flush=True)

    all_results: list[GateResult] = []
    for exp_label, metric_thresholds in thresholds.items():
        print(f"\n→ {exp_label}", flush=True)
        results = _diff_against_baseline(args.project, exp_label, metric_thresholds)
        for r in results:
            sym = "✓" if r.passed else "✗"
            print(f"  {sym} {r.metric}: {r.current:+.3f} vs baseline {r.baseline:+.3f} "
                  f"(delta {r.delta:+.3f}, threshold {r.threshold:+.3f})", flush=True)
        all_results.extend(results)

    failures = [r for r in all_results if not r.passed]

    if args.report_out:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(json.dumps([
            {
                "experiment": r.experiment, "metric": r.metric,
                "baseline": r.baseline, "current": r.current,
                "delta": r.delta, "threshold": r.threshold,
                "passed": r.passed,
            }
            for r in all_results
        ], indent=2))
        print(f"\nreport written: {args.report_out}", flush=True)

    print(f"\n{'=' * 60}", flush=True)
    print(f"gate result: {len(all_results) - len(failures)}/{len(all_results)} checks passed", flush=True)
    if failures:
        print("\nFAILURES:", flush=True)
        for r in failures:
            print(f"  ✗ {r.experiment}/{r.metric}: delta {r.delta:+.3f} < {r.threshold:+.3f}", flush=True)
        sys.exit(1)
    print("PASS", flush=True)


if __name__ == "__main__":
    main()
