"""Module entry point for ``python -m attestor.cli``.

Preserves the pre-split behavior where ``attestor/cli.py``'s
``if __name__ == "__main__": main()`` made the module directly executable.
"""

from __future__ import annotations

from attestor.cli.main import main

if __name__ == "__main__":
    main()
