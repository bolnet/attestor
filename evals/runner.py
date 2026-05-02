"""Matrix runner — read evals/matrix.yaml, run each cell as a Braintrust experiment.

Each cell becomes one invocation of ``evals/braintrust_longmemeval.py`` via
subprocess so each cell gets a clean Python process (own stack overrides, own
AgentMemory wiring, no cross-cell state leaks).

Two modes:

    --dry-run    Print the expanded cell list, total cost / wallclock estimate,
                 and exit. Always run this first when touching the matrix.
    --execute    Actually run the cells. Honors --filter <label-substring>
                 to scope to a subset.

CLI:

    set -a && source .env && set +a
    .venv/bin/python -m evals.runner --matrix evals/matrix.yaml --dry-run
    .venv/bin/python -m evals.runner --matrix evals/matrix.yaml --execute --filter GOLDEN
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class Cell:
    category: str
    label: str
    samples: int
    embedder_provider: str | None = None
    embedder_model: str | None = None
    self_consistency: bool = False
    cost_usd: float = 0.0
    wall_min: int = 0
    note: str = ""

    def cli_args(self) -> list[str]:
        args = [
            "--category", self.category,
            "--max-samples", str(self.samples),
            "--suffix", self.label,
        ]
        if self.embedder_provider:
            args.extend(["--embedder-provider", self.embedder_provider])
        if self.embedder_model:
            args.extend(["--embedder-model", self.embedder_model])
        if self.self_consistency:
            args.append("--self-consistency")
        return args


def load_matrix(path: Path) -> tuple[str, list[Cell]]:
    """Parse the matrix YAML; return (project, cells)."""
    with path.open() as f:
        spec = yaml.safe_load(f)
    project = spec.get("project", "attestor-lme-s")
    cells = []
    for raw in spec.get("cells", []):
        emb = raw.get("embedder") or {}
        cells.append(Cell(
            category=raw["category"],
            label=raw["label"],
            samples=int(raw.get("samples", 10)),
            embedder_provider=emb.get("provider"),
            embedder_model=emb.get("model"),
            self_consistency=bool(raw.get("self_consistency", False)),
            cost_usd=float(raw.get("cost_usd", 0.0)),
            wall_min=int(raw.get("wall_min", 0)),
            note=raw.get("note", ""),
        ))
    return project, cells


def filter_cells(cells: list[Cell], substring: str | None) -> list[Cell]:
    if not substring:
        return cells
    return [c for c in cells if substring in c.label or substring in c.category]


def print_plan(project: str, cells: list[Cell]) -> None:
    if not cells:
        print("(no cells)", flush=True)
        return
    total_cost = sum(c.cost_usd for c in cells)
    total_wall = sum(c.wall_min for c in cells)
    print(f"\nProject: {project}", flush=True)
    print(f"Cells:   {len(cells)}", flush=True)
    print(f"\n{'Category':<24} {'Label':<42} {'Samples':>8} {'$':>6} {'min':>5}", flush=True)
    print("─" * 92, flush=True)
    for c in cells:
        marker = "*" if c.note else " "
        print(f"{c.category:<24} {c.label:<42} {c.samples:>8} {c.cost_usd:>6.2f} {c.wall_min:>5}{marker}", flush=True)
    print("─" * 92, flush=True)
    print(f"{'TOTAL':<76} {total_cost:>6.2f} {total_wall:>5}", flush=True)
    notes = [c for c in cells if c.note]
    if notes:
        print("\nNotes:", flush=True)
        for c in notes:
            print(f"  * {c.label}: {c.note}", flush=True)


def run_cell(cell: Cell, *, dry_run: bool = False) -> int:
    """Execute one cell via subprocess to evals.braintrust_longmemeval.

    Returns the subprocess exit code (0 = success, non-zero = error).
    """
    cmd = [sys.executable, "-m", "evals.braintrust_longmemeval", *cell.cli_args()]
    print(f"\n→ {cell.category} :: {cell.label}", flush=True)
    print(f"  cmd: {' '.join(cmd)}", flush=True)
    if dry_run:
        return 0
    t0 = time.time()
    proc = subprocess.run(
        cmd,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).parent.parent)},
    )
    elapsed = time.time() - t0
    print(f"  exit={proc.returncode}  wallclock={elapsed:.0f}s", flush=True)
    return proc.returncode


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the LME-S eval matrix declared in evals/matrix.yaml.",
    )
    parser.add_argument("--matrix", type=Path, default=Path("evals/matrix.yaml"),
                        help="Path to the matrix YAML.")
    parser.add_argument("--filter", default=None,
                        help="Only run cells whose label or category contains this substring.")
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument("--dry-run", action="store_true",
                     help="Print the plan + total cost/wallclock; do NOT run cells.")
    grp.add_argument("--execute", action="store_true",
                     help="Run all (filtered) cells via braintrust_longmemeval.")
    args = parser.parse_args()

    if not args.dry_run and not args.execute:
        # Default to dry-run for safety.
        args.dry_run = True

    if not args.matrix.exists():
        raise SystemExit(f"matrix file not found: {args.matrix}")

    project, cells = load_matrix(args.matrix)
    cells = filter_cells(cells, args.filter)
    print_plan(project, cells)

    if args.dry_run:
        print("\n(dry-run; no cells executed)", flush=True)
        return

    if not os.environ.get("BRAINTRUST_API_KEY"):
        raise SystemExit("BRAINTRUST_API_KEY not set. Run: set -a && source .env && set +a")

    failures = 0
    for c in cells:
        if run_cell(c) != 0:
            failures += 1

    print("\n" + "=" * 60, flush=True)
    print(f"matrix done: {len(cells) - failures}/{len(cells)} cells succeeded", flush=True)
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
