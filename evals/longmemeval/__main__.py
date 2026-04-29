"""Allow ``python -m evals.longmemeval`` to invoke the runner CLI."""

from evals.longmemeval.runner import main

raise SystemExit(main())
