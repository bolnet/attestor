"""BEAM long-context benchmark runner (Phase 9.3, roadmap §G).

The v4 plan calls for "BEAM 1M token setting" — a long-context recall
benchmark stress-testing the memory layer at 10K → 1M token contexts.

This package ships the runner + scorer + summarize() shape, plus a
``DatasetLoader`` protocol so operators wire whichever long-context
dataset they want (RULER, BABILong, ∞Bench, HELMET, internal data) to
that interface. The default loader raises with a clear "wire your
dataset" message — we deliberately do not fabricate dataset URLs or
schemas the runner can't verify.
"""
