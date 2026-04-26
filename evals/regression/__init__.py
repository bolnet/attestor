"""Attestor regression suite — deterministic, no-LLM smoke gate (Phase 9.1, roadmap §G).

A YAML-driven catalog of recall cases. Each case ingests rounds, runs a
query, and asserts what the recalled ContextPack should (and must not)
contain. Designed to run in CI on every PR with a real Postgres but
no external API keys.
"""
