"""Allow ``python -m evals.knowledge_updates`` to invoke the runner CLI."""

from evals.knowledge_updates.runner import main

raise SystemExit(main())
