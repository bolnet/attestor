#!/usr/bin/env python3
"""
Benchmark: Claude Code with vs without memwright memory.

Literally installs/uninstalls memwright and runs real Claude Code CLI sessions.

- Baseline: No hooks, no MCP, no memory. Claude relies purely on context window.
- Memwright: Hooks installed (SessionStart injects memories, PostToolUse captures
  facts, Stop summarizes). MCP server available. Memory persists across compressions.

Measures: tokens, context %, cost, latency, recall accuracy, compression events.

Usage:
    # Baseline — no memory installed, run until context exhausts
    python bench_claude.py --mode baseline

    # Memwright — memory installed, same conversation
    python bench_claude.py --mode memwright

    # Both runs + comparison
    python bench_claude.py --mode both

    # Compare previous results
    python bench_claude.py --compare

    # Quick test
    python bench_claude.py --mode baseline --max-turns 15
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Conversation script
# ---------------------------------------------------------------------------

# Phase 1: Seed facts — told conversationally so Claude (and memwright hooks)
# can capture them naturally.
SEED_FACTS: List[tuple[str, List[str]]] = [
    (
        "Just so you know, my name is Priya Sharma and I'm a Staff Engineer at Stripe.",
        ["Priya", "Stripe", "Staff"],
    ),
    (
        "I live in Austin, Texas. Moved here from Seattle last year.",
        ["Austin", "Texas"],
    ),
    (
        "Our team is called Payments Core and we own the charge creation pipeline.",
        ["Payments Core", "charge"],
    ),
    (
        "We use Go 1.22 for all backend services, no exceptions.",
        ["Go", "1.22"],
    ),
    (
        "Our primary database is CockroachDB with 12 nodes across 3 regions.",
        ["CockroachDB", "12 nodes"],
    ),
    (
        "For message queues we use Apache Kafka with exactly-once semantics enabled.",
        ["Kafka", "exactly-once"],
    ),
    (
        "The frontend is Next.js 15 with React Server Components. TypeScript strict mode.",
        ["Next.js", "TypeScript"],
    ),
    (
        "Our CI runs on Buildkite with hermetic Bazel builds. Average build time is 8 minutes.",
        ["Buildkite", "Bazel"],
    ),
    (
        "I personally prefer Neovim with a custom Lua config. Been using it for 6 years.",
        ["Neovim", "Lua"],
    ),
    (
        "My direct manager is Alex Chen, VP of Engineering. He reports to the CTO.",
        ["Alex Chen", "VP"],
    ),
    (
        "We have a hard SLA of 99.99% uptime for the charge API. That's less than 52 minutes of downtime per year.",
        ["99.99", "52 minute"],
    ),
    (
        "Our on-call rotation is 1 week on, 5 weeks off. I'm on call next week starting Monday.",
        ["1 week", "5 week"],
    ),
    (
        "The team has 8 engineers: 5 backend, 2 frontend, 1 SRE. We're hiring 2 more backend engineers.",
        ["8 engineer", "5 backend"],
    ),
    (
        "For observability we use Datadog APM with custom dashboards. Alert threshold is p99 > 200ms.",
        ["Datadog", "p99", "200ms"],
    ),
    (
        "Our deployment pipeline: PR → Buildkite → staging (canary 5%) → production (blue-green).",
        ["canary", "blue-green"],
    ),
    (
        "I'm working on Project Mercury — migrating from synchronous to async charge processing. Deadline is April 15th.",
        ["Mercury", "async", "April 15"],
    ),
    (
        "The legacy charge system handles 50,000 TPS at peak. Mercury target is 200,000 TPS.",
        ["50,000", "200,000"],
    ),
    (
        "Our API versioning follows Stripe's date-based scheme: 2024-01-15, 2024-06-01, etc.",
        ["date-based", "2024"],
    ),
    (
        "For testing we use Go's built-in testing package plus testify for assertions. Coverage requirement is 85%.",
        ["testify", "85%"],
    ),
    (
        "Redis Cluster (6 shards) for rate limiting and idempotency keys. TTL is 24 hours.",
        ["Redis", "6 shard", "24 hour"],
    ),
    (
        "My side project is a CLI tool called 'paisa' for personal finance tracking in Rust.",
        ["paisa", "Rust"],
    ),
    (
        "I have a 3-year-old daughter named Aanya. She just started preschool at Bright Horizons.",
        ["Aanya", "daughter"],
    ),
    (
        "I run half-marathons. My PR is 1:38:22 from the Austin Half Marathon in February.",
        ["1:38", "half-marathon"],
    ),
    (
        "Our team's sprint is 2 weeks. Retro every other Friday at 3pm Central. Planning on Mondays.",
        ["2 week", "Friday", "Monday"],
    ),
    (
        "The feature flag system uses LaunchDarkly. We currently have 847 active flags.",
        ["LaunchDarkly", "847"],
    ),
    (
        "For secrets management we use HashiCorp Vault with auto-rotation every 30 days.",
        ["Vault", "30 day"],
    ),
    (
        "Our GraphQL gateway is Apollo Federation v2. 14 subgraphs, stitched at the edge.",
        ["Apollo", "Federation", "14 subgraph"],
    ),
    (
        "Load testing with k6. We run weekly soak tests: 10,000 concurrent users for 4 hours.",
        ["k6", "10,000", "4 hour"],
    ),
    (
        "I'm mentoring two junior engineers: Marcus (backend, joined 3 months ago) and Li Wei (frontend, 6 months).",
        ["Marcus", "Li Wei"],
    ),
    (
        "The incident response process: page → triage (15 min) → mitigate → RCA within 48 hours. We use PagerDuty.",
        ["PagerDuty", "48 hour", "RCA"],
    ),
]

# Phase 2: Filler — coding tasks that generate long responses to burn context fast.
FILLER_PROMPTS: List[str] = [
    "Write a Go function that implements a circuit breaker pattern with configurable thresholds for open/half-open/closed states. Include retry logic with exponential backoff and jitter.",
    "Explain the differences between optimistic and pessimistic locking in CockroachDB. Give me a detailed example of each with Go code using pgx driver.",
    "Write a comprehensive Kafka consumer in Go that handles exactly-once processing with idempotency keys stored in Redis. Include error handling, dead letter queue, and graceful shutdown.",
    "Design a rate limiter using Redis Cluster that supports sliding window with burst allowance. Write the full implementation in Go with Lua scripts for atomic operations.",
    "Write a Next.js 15 React Server Component that fetches data from a GraphQL gateway, handles streaming, and implements optimistic UI updates with TypeScript.",
    "Create a Buildkite pipeline YAML that runs hermetic Bazel builds with remote caching, parallel test execution, and automatic canary deployment to staging.",
    "Implement a distributed tracing middleware in Go that integrates with Datadog APM. Include context propagation across Kafka consumers, HTTP handlers, and gRPC.",
    "Write a Go service that handles graceful shutdown with in-flight request draining, Kafka consumer offset commits, CockroachDB connection pool cleanup, and Redis disconnect.",
    "Design a zero-downtime database migration strategy for CockroachDB. Write the migration runner in Go with rollback support, backfill logic, and progress tracking.",
    "Implement a Stripe-style date-based API versioning layer in Go. Include request/response transformation, deprecation headers, and version negotiation.",
    "Write a comprehensive test suite in Go using testify for a charge creation service. Include unit tests, integration tests with CockroachDB testcontainers, and Kafka consumer tests.",
    "Create a LaunchDarkly feature flag wrapper in Go that supports gradual rollouts, A/B testing, kill switches, and automatic cleanup for flags older than 90 days.",
    "Write a Prometheus metrics exporter in Go for a charge pipeline. Include histograms for latency, counters for throughput, gauges for queue depth, and custom collectors.",
    "Implement retry with jittered exponential backoff for payment processing. Handle transient vs permanent failures, circuit breaking, and DLQ routing. Full Go code.",
    "Design and implement a Redis Cluster caching layer with cache-aside, write-through for critical data, cache stampede protection, and TTL management. Go implementation.",
    "Write a Go HTTP middleware stack: request ID generation, structured logging with zerolog, panic recovery, CORS, JWT auth, rate limiting, and request/response logging.",
    "Implement a worker pool in Go for processing async charge events from Kafka. Include backpressure handling via channel buffering, metrics, graceful scaling, and health checks.",
    "Create a comprehensive error handling package in Go for payment services. Include error codes, sentinel errors, wrapping with context, stack traces, and Datadog integration.",
    "Write a Go implementation of the Saga pattern for distributed transactions across charge, ledger, notification, and audit services. Include compensating transactions.",
    "Design a health check system in Go that monitors CockroachDB, Kafka, Redis, and downstream services. Include liveness, readiness, startup probes, and degraded mode.",
    "Implement request coalescing (singleflight) in Go for charge status lookup. Multiple concurrent requests for the same charge ID should share a single DB query.",
    "Write a tamper-evident audit logging package in Go for charge mutations. Include HMAC checksums, log rotation, async batching, and compliance pipeline integration.",
    "Create a load shedding mechanism in Go that protects the charge API during spikes. Include priority queues, adaptive thresholds, and graceful degradation responses.",
    "Implement Server-Sent Events in Go for real-time charge event streaming. Handle reconnection with Last-Event-ID, event ordering, backpressure, and auth.",
    "Write a comprehensive Makefile for a Go monorepo: build, test, lint (golangci-lint), generate (protobuf, mocks), docker, deploy, local dev setup, and CI targets.",
    "Design an idempotency layer in Go for charge creation. Use Redis for key storage with proper TTL, race condition handling via Lua scripts, and response replay.",
    "Implement JSON Schema request validation middleware in Go. Include automatic schema generation from Go structs, detailed error messages, and schema caching.",
    "Write a canary deployment controller in Go that monitors Datadog metrics and automatically rolls back if p99 > 200ms or error rate > 0.1%. Include notification hooks.",
    "Create a Go package for CockroachDB connection pool management with automatic regional failover, health monitoring, query routing, and connection draining.",
    "Implement Consul-based service discovery in Go with service registration, TTL health checking, client-side load balancing (round-robin, least-connections), and retries.",
    "Write a Go benchmark suite for charge processing: memory allocation profiling, CPU profiling, comparison benchmarks with b.Run subtests, and CI integration.",
    "Design Kafka consumer message deduplication using counting Bloom filters with Redis backing. Include false positive rate management and periodic filter rotation.",
    "Create a multi-strategy rate limiter in Go: fixed window, sliding window log, sliding window counter, token bucket, and leaky bucket. Redis-backed with Lua atomics.",
    "Implement webhook delivery in Go with retry logic, exponential backoff with jitter, signature verification (HMAC-SHA256), dead letter management, and delivery tracking.",
    "Write a consistent hashing ring in Go for distributing charge processing across worker nodes. Include virtual nodes, smooth rebalancing, and bounded loads.",
    "Design a config management system in Go: Vault secrets with lease renewal, env vars, YAML files, live reloading via fsnotify, and typed config structs with validation.",
    "Implement batch charge processing in Go for CSV uploads. Include streaming CSV parsing, concurrent validation, progress tracking via SSE, error aggregation, and resumability.",
    "Write a structured logging package in Go with zerolog: JSON output, log levels, context fields, request-scoped loggers, sampling for hot paths, and Datadog integration.",
    "Create an API key management service in Go: key generation (Ed25519), scoping, per-key rate limits, rotation with grace period, usage tracking, and admin REST API.",
    "Implement a reverse proxy in Go with routing (chi), weighted round-robin load balancing, circuit breaking per backend, header manipulation, and request buffering.",
]

# Phase 3: Recall questions — interspersed to measure degradation over time.
RECALL_QUESTIONS: List[tuple[str, List[str], int]] = [
    ("What's my name and where do I work?", ["Priya", "Stripe"], 0),
    ("Where do I live?", ["Austin", "Texas"], 1),
    ("What's our team name?", ["Payments Core"], 2),
    ("What programming language do we use for backend?", ["Go"], 3),
    ("What database do we use and how many nodes?", ["CockroachDB", "12"], 4),
    ("What message queue do we use?", ["Kafka"], 5),
    ("What's our frontend tech stack?", ["Next.js", "TypeScript"], 6),
    ("What CI system do we use?", ["Buildkite"], 7),
    ("What text editor do I prefer?", ["Neovim"], 8),
    ("Who is my manager?", ["Alex Chen"], 9),
    ("What's our uptime SLA?", ["99.99"], 10),
    ("How does our on-call rotation work?", ["1 week", "5 week"], 11),
    ("How many engineers on the team?", ["8"], 12),
    ("What observability tool do we use?", ["Datadog"], 13),
    ("Describe our deployment pipeline.", ["canary", "blue-green"], 14),
    ("What's Project Mercury and when is it due?", ["Mercury", "async", "April"], 15),
    ("What TPS does legacy handle vs Mercury target?", ["50,000", "200,000"], 16),
    ("How do we version our API?", ["date-based"], 17),
    ("What test tools and coverage requirements?", ["testify", "85"], 18),
    ("What do we use Redis for?", ["Redis", "rate limit"], 19),
    ("What's my side project?", ["paisa", "Rust"], 20),
    ("Tell me about my family.", ["Aanya", "daughter"], 21),
    ("What's my running PR?", ["1:38"], 22),
    ("When are sprint ceremonies?", ["Friday", "Monday"], 23),
    ("What feature flag system?", ["LaunchDarkly"], 24),
    ("How do we manage secrets?", ["Vault"], 25),
    ("What's our GraphQL setup?", ["Apollo", "Federation"], 26),
    ("How do we load test?", ["k6"], 27),
    ("Who am I mentoring?", ["Marcus", "Li Wei"], 28),
    ("What's our incident response process?", ["PagerDuty", "RCA"], 29),
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class TurnResult:
    turn_number: int
    phase: str  # "seed", "filler", "recall"
    prompt_preview: str
    response_preview: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0
    total_cost_usd: float = 0.0
    duration_ms: int = 0
    duration_api_ms: int = 0
    context_window: int = 200_000
    effective_context_tokens: int = 0
    context_pct: float = 0.0
    recall_expected: List[str] = field(default_factory=list)
    recall_found: List[str] = field(default_factory=list)
    recall_accuracy: float = 0.0
    compression_detected: bool = False


@dataclass
class BenchmarkRun:
    mode: str
    model: str
    started_at: str = ""
    finished_at: str = ""
    work_dir: str = ""
    turns: List[TurnResult] = field(default_factory=list)
    total_cost_usd: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    compression_count: int = 0
    recall_scores: List[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Install / Uninstall memwright (system-wide + project-level)
# ---------------------------------------------------------------------------


def _check_system_memwright() -> Dict[str, Any]:
    """Check all places memwright could be installed system-wide."""
    checks: Dict[str, Any] = {
        "memwright_bin": shutil.which("memwright"),
        "agent_memory_bin": shutil.which("agent-memory"),
        "pipx_installed": False,
        "pip_installed": False,
        "user_hooks": [],
        "user_mcp": [],
    }

    # Check pipx
    try:
        result = subprocess.run(
            ["pipx", "list", "--short"], capture_output=True, text=True, timeout=10,
        )
        if "memwright" in result.stdout.lower():
            checks["pipx_installed"] = True
    except Exception:
        pass

    # Check pip (system python, NOT the local .venv which is the dev environment)
    # We only care if memwright is installed globally where Claude Code could find it.
    # Skip this check — PATH and pipx checks are sufficient.

    # Check user-level Claude Code settings for memwright hooks
    user_settings = os.path.expanduser("~/.claude/settings.json")
    if os.path.exists(user_settings):
        try:
            with open(user_settings) as f:
                data = json.load(f)
            hooks = data.get("hooks", {})
            for event_type, hook_list in hooks.items():
                for hook_group in hook_list:
                    for hook in hook_group.get("hooks", []):
                        cmd = hook.get("command", "")
                        if "memwright" in cmd or "agent-memory" in cmd:
                            checks["user_hooks"].append({
                                "event": event_type,
                                "command": cmd,
                            })
        except Exception:
            pass

    # Check user-level MCP config
    for mcp_path in [
        os.path.expanduser("~/.claude/settings.json"),
        os.path.expanduser("~/.claude.json"),
    ]:
        if os.path.exists(mcp_path):
            try:
                with open(mcp_path) as f:
                    data = json.load(f)
                servers = data.get("mcpServers", {})
                for name, cfg in servers.items():
                    cmd = cfg.get("command", "")
                    if "memwright" in cmd or "agent-memory" in cmd:
                        checks["user_mcp"].append({"name": name, "config": mcp_path})
            except Exception:
                pass

    return checks


def verify_memwright_not_installed() -> None:
    """Verify memwright is completely absent from the system. Error if found."""
    print("\n  Checking system for memwright installation...")
    checks = _check_system_memwright()

    problems = []

    if checks["memwright_bin"]:
        problems.append(f"  - 'memwright' binary found at: {checks['memwright_bin']}")
        problems.append(f"    Fix: pipx uninstall memwright")

    if checks["agent_memory_bin"]:
        problems.append(f"  - 'agent-memory' binary found at: {checks['agent_memory_bin']}")
        problems.append(f"    Fix: pipx uninstall memwright  OR  pip uninstall memwright")

    if checks["pipx_installed"]:
        problems.append(f"  - memwright installed via pipx")
        problems.append(f"    Fix: pipx uninstall memwright")

    for hook in checks["user_hooks"]:
        problems.append(f"  - User-level hook found: [{hook['event']}] {hook['command']}")
        problems.append(f"    Fix: Remove memwright hooks from ~/.claude/settings.json")

    for mcp in checks["user_mcp"]:
        problems.append(f"  - User-level MCP server '{mcp['name']}' in {mcp['config']}")
        problems.append(f"    Fix: Remove memwright MCP from {mcp['config']}")

    if problems:
        print("\n  MEMWRIGHT DETECTED! Baseline requires clean system.\n")
        for p in problems:
            print(p)
        print("\n  Run the following to uninstall:")
        print("    pipx uninstall memwright 2>/dev/null")
        print("    pip uninstall memwright -y 2>/dev/null")
        print("    # Remove memwright hooks/MCP from ~/.claude/settings.json")
        print()
        sys.exit(1)

    print("  OK: memwright not found in PATH, pipx, user hooks, or user MCP.")


def verify_memwright_installed() -> str:
    """Verify memwright is installed system-wide. Install if missing. Return binary path."""
    print("\n  Checking system for memwright installation...")
    checks = _check_system_memwright()

    memwright_bin = checks["memwright_bin"] or checks["agent_memory_bin"]

    if not memwright_bin:
        print("  memwright not found in PATH. Installing via pipx...")
        result = subprocess.run(
            ["pipx", "install", "memwright"],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            print(f"  pipx install failed: {result.stderr[:300]}")
            print("  Trying pip install...")
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "memwright"],
                capture_output=True, text=True, timeout=120,
            )
            if result.returncode != 0:
                print(f"  pip install failed: {result.stderr[:300]}")
                sys.exit(1)

        memwright_bin = shutil.which("memwright") or shutil.which("agent-memory")
        if not memwright_bin:
            print("  ERROR: memwright installed but binary not found in PATH.")
            sys.exit(1)

    print(f"  OK: memwright binary at {memwright_bin}")
    return memwright_bin


def install_memwright(work_dir: str) -> None:
    """Install memwright: system-wide binary + project-level hooks + MCP + memory store."""
    memwright_bin = verify_memwright_installed()

    # 1. Create .memwright store and seed with facts
    store_path = os.path.join(work_dir, ".memwright")
    os.makedirs(store_path, exist_ok=True)

    from agent_memory.core import AgentMemory
    mem = AgentMemory(store_path)

    # Seed all 30 facts (simulates memories accumulated from prior sessions)
    print(f"  [install] Seeding {len(SEED_FACTS)} memories into store...")
    for content, keywords in SEED_FACTS:
        tags = [kw.lower().replace(" ", "-") for kw in keywords[:3]]
        entity = None
        if any(kw in content for kw in ["Priya", "my ", "I ", "I'"]):
            entity = "Priya"
        elif any(kw in content for kw in ["team", "Team"]):
            entity = "Payments Core"
        elif "Mercury" in content:
            entity = "Project Mercury"
        mem.add(content=content, tags=tags, category="general", entity=entity)

    stats = mem.stats()
    print(f"  [install] Memory store: {store_path} ({stats.get('total_memories', '?')} memories)")
    mem.close()

    # 2. Project-level hooks in .claude/settings.json
    claude_dir = os.path.join(work_dir, ".claude")
    os.makedirs(claude_dir, exist_ok=True)

    settings = {
        "hooks": {
            "SessionStart": [
                {
                    "matcher": "startup|resume|clear|compact",
                    "hooks": [
                        {
                            "type": "command",
                            "command": f"{memwright_bin} hook session-start",
                            "timeout": 10,
                        }
                    ],
                }
            ],
            "PostToolUse": [
                {
                    "matcher": "Write|Edit|Bash",
                    "hooks": [
                        {
                            "type": "command",
                            "command": f"{memwright_bin} hook post-tool-use",
                            "timeout": 5,
                        }
                    ],
                }
            ],
            "Stop": [
                {
                    "hooks": [
                        {
                            "type": "command",
                            "command": f"{memwright_bin} hook stop",
                            "timeout": 10,
                        }
                    ],
                }
            ],
        }
    }

    settings_path = os.path.join(claude_dir, "settings.json")
    with open(settings_path, "w") as f:
        json.dump(settings, f, indent=2)
    print(f"  [install] Hooks: {settings_path}")

    # 3. MCP server in .mcp.json
    mcp_config = {
        "mcpServers": {
            "memwright": {
                "command": memwright_bin,
                "args": ["mcp", "--path", store_path],
            }
        }
    }
    mcp_path = os.path.join(work_dir, ".mcp.json")
    with open(mcp_path, "w") as f:
        json.dump(mcp_config, f, indent=2)
    print(f"  [install] MCP: {mcp_path}")


def uninstall_memwright(work_dir: str) -> None:
    """Remove ALL memwright artifacts from the working directory."""
    removed = []

    # Remove .memwright store
    store_path = os.path.join(work_dir, ".memwright")
    if os.path.exists(store_path):
        shutil.rmtree(store_path)
        removed.append(".memwright/")

    # Remove .mcp.json
    mcp_path = os.path.join(work_dir, ".mcp.json")
    if os.path.exists(mcp_path):
        os.remove(mcp_path)
        removed.append(".mcp.json")

    # Remove project-level .claude/settings.json (only if it has memwright hooks)
    settings_path = os.path.join(work_dir, ".claude", "settings.json")
    if os.path.exists(settings_path):
        try:
            with open(settings_path) as f:
                data = json.load(f)
            # Check if it has memwright hooks
            hooks_str = json.dumps(data.get("hooks", {}))
            if "memwright" in hooks_str or "agent-memory" in hooks_str:
                os.remove(settings_path)
                removed.append(".claude/settings.json")
        except Exception:
            pass

    if removed:
        print(f"  [uninstall] Removed: {', '.join(removed)}")
    else:
        print(f"  [uninstall] Nothing to remove (already clean)")


# ---------------------------------------------------------------------------
# Working directory setup
# ---------------------------------------------------------------------------


def setup_work_dir(mode: str) -> str:
    """Create a temp working directory that looks like a real project.

    For baseline: verifies memwright is NOT installed system-wide.
    For memwright: installs memwright system-wide + project hooks + seeds memories.
    """
    work_dir = tempfile.mkdtemp(prefix=f"bench_{mode}_")

    # Initialize git repo (Claude Code expects one)
    subprocess.run(["git", "init", "-q"], cwd=work_dir, capture_output=True)
    subprocess.run(
        ["git", "commit", "--allow-empty", "-m", "init", "-q"],
        cwd=work_dir,
        capture_output=True,
        env={**os.environ, "GIT_AUTHOR_NAME": "bench", "GIT_AUTHOR_EMAIL": "bench@test",
             "GIT_COMMITTER_NAME": "bench", "GIT_COMMITTER_EMAIL": "bench@test"},
    )

    # Create a minimal CLAUDE.md
    with open(os.path.join(work_dir, "CLAUDE.md"), "w") as f:
        f.write("# Benchmark Project\nThis is a benchmark workspace. Answer questions concisely.\n")

    # Create a dummy Go project for coding context
    os.makedirs(os.path.join(work_dir, "cmd"), exist_ok=True)
    with open(os.path.join(work_dir, "cmd", "main.go"), "w") as f:
        f.write('package main\n\nimport "fmt"\n\nfunc main() {\n\tfmt.Println("hello")\n}\n')

    with open(os.path.join(work_dir, "go.mod"), "w") as f:
        f.write("module github.com/example/charge-service\n\ngo 1.22\n")

    if mode == "baseline":
        # CRITICAL: verify memwright is nowhere on this system
        verify_memwright_not_installed()
        uninstall_memwright(work_dir)  # clean project dir just in case
    elif mode == "memwright":
        # Install memwright system-wide + project hooks + seed memories
        install_memwright(work_dir)

    return work_dir


# ---------------------------------------------------------------------------
# Claude CLI runner
# ---------------------------------------------------------------------------


def run_claude_turn(
    prompt: str,
    session_id: Optional[str],
    model: str,
    work_dir: str,
) -> Dict[str, Any]:
    """Run a single Claude CLI turn and return parsed JSON result."""
    cmd = [
        "claude",
        "-p", prompt,
        "--output-format", "json",
        "--model", model,
    ]

    if session_id:
        cmd.extend(["--resume", session_id])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            cwd=work_dir,
        )

        if result.returncode != 0:
            return {"error": result.stderr[:500] or "non-zero exit", "raw": result.stdout[:500]}

        return json.loads(result.stdout)

    except subprocess.TimeoutExpired:
        return {"error": "timeout after 300s"}
    except json.JSONDecodeError:
        return {"error": "invalid JSON", "raw": result.stdout[:500] if result else ""}
    except Exception as e:
        return {"error": str(e)}


def extract_tokens(data: Dict[str, Any]) -> Dict[str, int]:
    """Extract token counts from claude CLI JSON output."""
    model_usage = data.get("modelUsage", {})
    total_input = 0
    total_output = 0
    total_cache_create = 0
    total_cache_read = 0
    context_window = 200_000

    for model_data in model_usage.values():
        total_input += model_data.get("inputTokens", 0)
        total_output += model_data.get("outputTokens", 0)
        total_cache_create += model_data.get("cacheCreationInputTokens", 0)
        total_cache_read += model_data.get("cacheReadInputTokens", 0)
        context_window = model_data.get("contextWindow", context_window)

    return {
        "input_tokens": total_input,
        "output_tokens": total_output,
        "cache_creation_tokens": total_cache_create,
        "cache_read_tokens": total_cache_read,
        "context_window": context_window,
    }


def score_recall(response: str, expected_keywords: List[str]) -> tuple[List[str], float]:
    """Score recall: fraction of expected keywords found in response."""
    response_lower = response.lower()
    found = [kw for kw in expected_keywords if kw.lower() in response_lower]
    accuracy = len(found) / len(expected_keywords) if expected_keywords else 0.0
    return found, accuracy


# ---------------------------------------------------------------------------
# Build turn schedule
# ---------------------------------------------------------------------------


def build_turn_schedule(max_turns: int) -> List[Dict[str, Any]]:
    """Build interleaved seed → filler+recall schedule.

    - Turns 1-30: Seed facts
    - Turns 31+: 4 filler, 3 recall, repeat
    """
    schedule: List[Dict[str, Any]] = []

    # Seed phase
    for msg, keywords in SEED_FACTS:
        if len(schedule) >= max_turns:
            break
        schedule.append({"phase": "seed", "prompt": msg, "recall_keywords": keywords})

    # Filler + recall interleaved
    filler_idx = 0
    recall_idx = 0

    while len(schedule) < max_turns:
        # 4 filler turns
        for _ in range(4):
            if len(schedule) >= max_turns:
                break
            schedule.append({
                "phase": "filler",
                "prompt": FILLER_PROMPTS[filler_idx % len(FILLER_PROMPTS)],
                "recall_keywords": [],
            })
            filler_idx += 1

        # 3 recall turns
        for _ in range(3):
            if len(schedule) >= max_turns or recall_idx >= len(RECALL_QUESTIONS):
                break
            question, keywords, fact_idx = RECALL_QUESTIONS[recall_idx]
            schedule.append({
                "phase": "recall",
                "prompt": question,
                "recall_keywords": keywords,
                "fact_index": fact_idx,
            })
            recall_idx += 1

        # Cycle recall questions if exhausted
        if recall_idx >= len(RECALL_QUESTIONS):
            recall_idx = 0

    return schedule[:max_turns]


# ---------------------------------------------------------------------------
# Main benchmark runner
# ---------------------------------------------------------------------------


def run_benchmark(
    mode: str,
    model: str = "haiku",
    max_turns: int = 200,
    output_dir: str = "benchmark-logs",
) -> BenchmarkRun:
    """Run the full benchmark."""

    run_data = BenchmarkRun(
        mode=mode,
        model=model,
        started_at=datetime.now().isoformat(),
    )

    # Setup working directory with or without memwright
    print(f"\n  Setting up {mode} working directory...")
    work_dir = setup_work_dir(mode)
    run_data.work_dir = work_dir
    print(f"  Work dir: {work_dir}")

    schedule = build_turn_schedule(max_turns)
    session_id: Optional[str] = None
    prev_effective_context = 0

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    results_file = os.path.join(output_dir, f"bench-{mode}-{model}-{timestamp}.json")

    print(f"\n{'='*90}")
    print(f"  BENCHMARK: {mode.upper()} | Model: {model} | Max turns: {max_turns}")
    print(f"  Memory: {'INSTALLED (hooks + MCP)' if mode == 'memwright' else 'NOT INSTALLED'}")
    print(f"  Output: {results_file}")
    print(f"{'='*90}\n")

    print(f"{'#':>4} {'Phase':>7} {'InTok':>8} {'OutTok':>7} {'CachCr':>8} "
          f"{'CachRd':>8} {'EffCtx':>9} {'Ctx%':>6} {'Cost':>7} {'Time':>6} {'Recall':>8}")
    print("-" * 95)

    try:
        for i, turn_info in enumerate(schedule):
            turn_num = i + 1
            phase = turn_info["phase"]
            prompt = turn_info["prompt"]
            recall_keywords = turn_info.get("recall_keywords", [])

            # Run the turn
            result = run_claude_turn(prompt, session_id, model, work_dir)

            if "error" in result:
                print(f"\n  ERROR turn {turn_num}: {result['error'][:200]}")
                if "raw" in result:
                    print(f"  raw: {result['raw'][:200]}")
                # Don't break — try to continue
                continue

            # Extract session ID
            if not session_id:
                session_id = result.get("session_id")

            # Metrics
            tokens = extract_tokens(result)
            cost = result.get("total_cost_usd", 0.0)
            duration_ms = result.get("duration_ms", 0)
            duration_api_ms = result.get("duration_api_ms", 0)
            response_text = result.get("result", "")

            # Effective context = what was actually sent to the API
            effective_context = (
                tokens["input_tokens"]
                + tokens["cache_creation_tokens"]
                + tokens["cache_read_tokens"]
            )
            context_pct = (effective_context / tokens["context_window"]) * 100 if tokens["context_window"] else 0

            # Detect compression: effective context drops significantly
            compression = False
            if turn_num > 5 and prev_effective_context > 0:
                if effective_context < prev_effective_context * 0.6:
                    compression = True
                    run_data.compression_count += 1
            prev_effective_context = effective_context

            # Score recall
            found, accuracy = score_recall(response_text, recall_keywords)

            turn_result = TurnResult(
                turn_number=turn_num,
                phase=phase,
                prompt_preview=prompt[:80],
                response_preview=response_text[:150],
                input_tokens=tokens["input_tokens"],
                output_tokens=tokens["output_tokens"],
                cache_creation_tokens=tokens["cache_creation_tokens"],
                cache_read_tokens=tokens["cache_read_tokens"],
                total_cost_usd=cost,
                duration_ms=duration_ms,
                duration_api_ms=duration_api_ms,
                context_window=tokens["context_window"],
                effective_context_tokens=effective_context,
                context_pct=round(context_pct, 1),
                recall_expected=recall_keywords,
                recall_found=found,
                recall_accuracy=round(accuracy, 3),
                compression_detected=compression,
            )

            run_data.turns.append(turn_result)
            run_data.total_cost_usd += cost
            run_data.total_input_tokens += tokens["input_tokens"]
            run_data.total_output_tokens += tokens["output_tokens"]
            if phase == "recall":
                run_data.recall_scores.append(accuracy)

            # Print row
            recall_str = f"{accuracy*100:5.1f}%" if phase == "recall" else "     -"
            marker = " <<< COMPRESSED" if compression else ""
            print(
                f"{turn_num:>4} {phase:>7} "
                f"{tokens['input_tokens']:>8,} {tokens['output_tokens']:>7,} "
                f"{tokens['cache_creation_tokens']:>8,} {tokens['cache_read_tokens']:>8,} "
                f"{effective_context:>9,} {context_pct:>5.1f}% "
                f"${cost:>6.3f} {duration_ms/1000:>5.1f}s "
                f"{recall_str}{marker}"
            )

            # Save every 10 turns
            if turn_num % 10 == 0:
                _save_results(run_data, results_file)

    except KeyboardInterrupt:
        print("\n\n  Interrupted by user.")

    finally:
        run_data.finished_at = datetime.now().isoformat()
        _save_results(run_data, results_file)
        print(f"\n  Results saved: {results_file}")
        print(f"  Work dir preserved: {work_dir}")

    _print_summary(run_data)
    return run_data


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _save_results(run_data: BenchmarkRun, filepath: str) -> None:
    """Save results to JSON."""
    data = {
        "mode": run_data.mode,
        "model": run_data.model,
        "started_at": run_data.started_at,
        "finished_at": run_data.finished_at,
        "work_dir": run_data.work_dir,
        "total_turns": len(run_data.turns),
        "total_cost_usd": round(run_data.total_cost_usd, 4),
        "total_input_tokens": run_data.total_input_tokens,
        "total_output_tokens": run_data.total_output_tokens,
        "compression_count": run_data.compression_count,
        "recall_scores": run_data.recall_scores,
        "avg_recall_accuracy": (
            round(sum(run_data.recall_scores) / len(run_data.recall_scores), 3)
            if run_data.recall_scores else 0.0
        ),
        "turns": [asdict(t) for t in run_data.turns],
    }
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


def _print_summary(run_data: BenchmarkRun) -> None:
    """Print run summary."""
    print(f"\n{'='*90}")
    print(f"  SUMMARY: {run_data.mode.upper()}")
    print(f"{'='*90}")
    print(f"  Turns completed:     {len(run_data.turns)}")
    print(f"  Total cost:          ${run_data.total_cost_usd:.4f}")
    print(f"  Total input tokens:  {run_data.total_input_tokens:,}")
    print(f"  Total output tokens: {run_data.total_output_tokens:,}")
    print(f"  Compressions:        {run_data.compression_count}")

    if run_data.recall_scores:
        avg = sum(run_data.recall_scores) / len(run_data.recall_scores)
        print(f"  Recall accuracy:     {avg*100:.1f}% (avg over {len(run_data.recall_scores)} questions)")

        n = len(run_data.recall_scores)
        if n >= 6:
            early = run_data.recall_scores[:n//3]
            mid = run_data.recall_scores[n//3:2*n//3]
            late = run_data.recall_scores[2*n//3:]
            print(f"    Early ({len(early):>2} qs): {sum(early)/len(early)*100:.1f}%")
            print(f"    Mid   ({len(mid):>2} qs): {sum(mid)/len(mid)*100:.1f}%")
            print(f"    Late  ({len(late):>2} qs): {sum(late)/len(late)*100:.1f}%")

    if run_data.turns:
        peak = max(t.context_pct for t in run_data.turns)
        total_ms = sum(t.duration_ms for t in run_data.turns)
        print(f"  Peak context %:      {peak:.1f}%")
        print(f"  Total wall time:     {total_ms/1000:.0f}s ({total_ms/60000:.1f}min)")

    print(f"{'='*90}\n")


def compare_runs(baseline_path: str, memwright_path: str) -> None:
    """Compare two result files side by side."""
    with open(baseline_path) as f:
        b = json.load(f)
    with open(memwright_path) as f:
        m = json.load(f)

    print(f"\n{'='*75}")
    print(f"  COMPARISON: Baseline vs Memwright")
    print(f"{'='*75}")
    print(f"  {'Metric':<30} {'Baseline':>20} {'Memwright':>20}")
    print(f"  {'-'*70}")

    rows = [
        ("Turns", b["total_turns"], m["total_turns"]),
        ("Total cost", f"${b['total_cost_usd']:.4f}", f"${m['total_cost_usd']:.4f}"),
        ("Input tokens", f"{b['total_input_tokens']:,}", f"{m['total_input_tokens']:,}"),
        ("Output tokens", f"{b['total_output_tokens']:,}", f"{m['total_output_tokens']:,}"),
        ("Compressions", b["compression_count"], m["compression_count"]),
        ("Avg recall accuracy", f"{b['avg_recall_accuracy']*100:.1f}%", f"{m['avg_recall_accuracy']*100:.1f}%"),
    ]
    for label, bv, mv in rows:
        print(f"  {label:<30} {str(bv):>20} {str(mv):>20}")

    # Degradation curves
    for name, data in [("Baseline", b), ("Memwright", m)]:
        scores = data.get("recall_scores", [])
        if len(scores) >= 6:
            n = len(scores)
            early = scores[:n//3]
            mid = scores[n//3:2*n//3]
            late = scores[2*n//3:]
            print(f"\n  {name} recall degradation:")
            print(f"    Early: {sum(early)/len(early)*100:.1f}%  "
                  f"Mid: {sum(mid)/len(mid)*100:.1f}%  "
                  f"Late: {sum(late)/len(late)*100:.1f}%")

    # Token efficiency
    if b["total_turns"] > 0 and m["total_turns"] > 0:
        b_per_turn = b["total_input_tokens"] / b["total_turns"]
        m_per_turn = m["total_input_tokens"] / m["total_turns"]
        savings = ((b_per_turn - m_per_turn) / b_per_turn) * 100 if b_per_turn > 0 else 0
        print(f"\n  Token efficiency:")
        print(f"    Baseline avg input/turn:  {b_per_turn:,.0f}")
        print(f"    Memwright avg input/turn: {m_per_turn:,.0f}")
        print(f"    Savings:                  {savings:.1f}%")

    print(f"\n{'='*75}\n")


def find_latest(output_dir: str, mode: str) -> Optional[str]:
    """Find most recent results file for a mode."""
    if not os.path.isdir(output_dir):
        return None
    files = sorted(
        [f for f in os.listdir(output_dir) if f.startswith(f"bench-{mode}-") and f.endswith(".json")],
        reverse=True,
    )
    return os.path.join(output_dir, files[0]) if files else None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Claude Code with vs without memwright memory"
    )
    parser.add_argument(
        "--mode", choices=["baseline", "memwright", "both"], default="baseline",
        help="baseline=no memory, memwright=hooks+MCP installed, both=run both then compare",
    )
    parser.add_argument("--model", default="haiku", help="Model (default: haiku)")
    parser.add_argument("--max-turns", type=int, default=200, help="Max turns (default: 200)")
    parser.add_argument("--output-dir", default="benchmark-logs", help="Results dir")
    parser.add_argument("--compare", action="store_true", help="Compare latest results")
    parser.add_argument("--compare-files", nargs=2, metavar=("BASELINE", "MEMWRIGHT"))

    args = parser.parse_args()

    if args.compare or args.compare_files:
        if args.compare_files:
            compare_runs(*args.compare_files)
        else:
            bp = find_latest(args.output_dir, "baseline")
            mp = find_latest(args.output_dir, "memwright")
            if not bp or not mp:
                print(f"Missing results. Baseline: {bp or 'NONE'}  Memwright: {mp or 'NONE'}")
                sys.exit(1)
            compare_runs(bp, mp)
        return

    if args.mode == "both":
        # Run both: baseline first (requires memwright uninstalled),
        # then install memwright and run the memwright benchmark.
        print("\n" + "=" * 90)
        print("  STEP 1/3: BASELINE RUN (memwright must NOT be installed)")
        print("=" * 90)
        run_benchmark(mode="baseline", model=args.model, max_turns=args.max_turns, output_dir=args.output_dir)

        print("\n" + "=" * 90)
        print("  STEP 2/3: MEMWRIGHT RUN (installing memwright, hooks + MCP)")
        print("=" * 90)
        run_benchmark(mode="memwright", model=args.model, max_turns=args.max_turns, output_dir=args.output_dir)

        print("\n" + "=" * 90)
        print("  STEP 3/3: COMPARISON")
        print("=" * 90)
        bp = find_latest(args.output_dir, "baseline")
        mp = find_latest(args.output_dir, "memwright")
        if bp and mp:
            compare_runs(bp, mp)
        else:
            print("  Could not find both result files for comparison.")

    elif args.mode == "baseline":
        print("\n>>> BASELINE RUN (memwright must NOT be installed) <<<")
        run_benchmark(mode="baseline", model=args.model, max_turns=args.max_turns, output_dir=args.output_dir)

    elif args.mode == "memwright":
        print("\n>>> MEMWRIGHT RUN (installing memwright — hooks + MCP) <<<")
        run_benchmark(mode="memwright", model=args.model, max_turns=args.max_turns, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
