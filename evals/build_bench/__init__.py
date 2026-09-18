# SPDX-FileCopyrightText: 2026 Surendra Singh <66422685+bolnet@users.noreply.github.com>
# SPDX-License-Identifier: MIT
"""Build-bench: a live software-build token-savings benchmark.

Drive a real multi-agent software build (via the Claude Code Workflow tool) over
many sequential steps. At each step attestor recalls only the relevant prior
decisions instead of re-feeding the whole accumulated history. The ledger records
the *single-run counterfactual*: recall-packed tokens vs the full-history tokens a
memoryless agent would have needed at that same step.
"""
