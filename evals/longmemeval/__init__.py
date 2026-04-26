"""LongMemEval runner — thin wrapper over attestor.longmemeval (Phase 9.2).

The heavy lifting (ingest → answer → judge → score) lives in
``attestor.longmemeval``. This package provides:

  - A canonical CLI entry (``python -m evals.longmemeval``)
  - A summary extractor that maps LMERunReport → BenchmarkSummary
    (the standard shape every eval runner produces, used by the CI gate)
"""
