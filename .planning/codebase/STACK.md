# Technology Stack

**Analysis Date:** 2026-03-15

## Languages

**Primary:**
- Python 3.9+ - All application code, CLI tools, benchmarks

**Secondary:**
- SQL - PostgreSQL schemas and Neo4j Cypher queries

## Runtime

**Environment:**
- Python 3.9, 3.10, 3.11, 3.12, 3.13 (supported)

**Package Manager:**
- pip with hatchling build backend
- Lockfile: pyproject.toml (PEP 517 format)

## Frameworks

**Core:**
- SQLite - Embedded local storage (no external dependency)
- PostgreSQL + pgvector - Vector semantic search via Docker
- Neo4j - Entity relationship graph via Docker

**CLI:**
- argparse - Built-in Python CLI argument parsing

**Testing:**
- pytest 7.0+ - Test runner
- pytest-cov 4.0+ - Coverage reporting

**Build/Dev:**
- hatchling - Python package builder
- GitHub Actions - CI/CD for PyPI publishing

## Key Dependencies

**Critical:**
- psycopg[binary] 3.2.0+ - PostgreSQL driver with binary support
- psycopg-pool 3.2.0+ - Connection pooling for PostgreSQL
- pgvector 0.3.0+ - pgvector extension client for vector operations
- neo4j 5.0+ - Neo4j database driver
- numpy 1.24.0+ - Array operations for embeddings (1536-dimensional vectors)
- openai 1.0.0+ - OpenAI/OpenRouter API client for embeddings

**Infrastructure:**
- mcp 1.0.0+ - Model Context Protocol server (optional, for Claude Code/Cursor integration)

## Optional Dependencies

**Vectors (pgvector/PostgreSQL):**
- Install with: `pip install "memwright[vectors]"`
- Includes: psycopg[binary], psycopg-pool, pgvector, numpy, openai

**Neo4j:**
- Install with: `pip install "memwright[neo4j]"`
- Includes: neo4j

**Extraction (LLM-based entity/memory extraction):**
- Install with: `pip install "memwright[extraction]"`
- Includes: openai

**MCP (Claude Code/Cursor integration):**
- Install with: `pip install "memwright[mcp]"`
- Includes: mcp

**All (Production):**
- Install with: `pip install "memwright[all]"`
- Includes all optional dependencies

**Development:**
- Install with: `pip install "memwright[dev]"`
- Includes: pytest, pytest-cov

## Configuration

**Environment Variables:**
- `OPENROUTER_API_KEY` - OpenRouter API key (preferred for embeddings)
- `OPENAI_API_KEY` - OpenAI API key (fallback for embeddings)
- `PG_CONNECTION_STRING` - PostgreSQL connection URL (default: `postgresql://memwright:memwright@localhost:5432/memwright`)
- `NEO4J_URI` - Neo4j Bolt URL (default: `bolt://localhost:7687`)
- `NEO4J_USER` - Neo4j username (default: `neo4j`)
- `NEO4J_PASSWORD` - Neo4j password (default: `memwright`)
- `NEO4J_DATABASE` - Neo4j database name (default: `neo4j`)
- `ANTHROPIC_API_KEY` - Anthropic API (optional, for future LLM use)

**Configuration Files:**
- `config.json` - Per-memory-store configuration (auto-created at `{store_path}/config.json`)
- `pyproject.toml` - Package metadata and dependencies
- `docker-compose.yml` - Docker services for PostgreSQL/pgvector and Neo4j

## Platform Requirements

**Development:**
- Docker Desktop - Required for PostgreSQL and Neo4j containers
- Python 3.9+ with pip or pipx (macOS Homebrew users prefer pipx)
- `.env` file with embedding API key (OPENROUTER_API_KEY or OPENAI_API_KEY)

**Production:**
- Docker (PostgreSQL 16 with pgvector extension, Neo4j 5 Community or Enterprise)
- Python 3.9+ runtime
- Environment variables for all secrets (embedding API keys, database passwords)
- Embedding API access (OpenRouter or OpenAI)

## Entry Points

**CLI Scripts:**
- `agent-memory` - Main CLI (defined in pyproject.toml `[project.scripts]`)
- `memwright` - Alias for `agent-memory`

**MCP Server:**
- Exposed via `agent_memory.mcp.server:create_server()` - For Claude Code / Cursor integration
- Run as: `python -m agent_memory.mcp server {memory_path}`

**Python API:**
- `from agent_memory import AgentMemory` - Main class in `agent_memory/core.py`

## Storage Locations

**Local SQLite (always available):**
- `{memory_path}/memory.db` - Core SQLite database

**Configuration:**
- `{memory_path}/config.json` - Per-store configuration

**Docker Services (via docker-compose.yml):**
- PostgreSQL: `localhost:5432` (container: memwright-postgres)
- Neo4j: `localhost:7687` (Bolt), `localhost:7474` (HTTP) (container: memwright-neo4j)
- Neo4j Test: `localhost:7688` (Bolt test instance)

## Packaging & Publishing

**PyPI Package:**
- Name: `memwright`
- Version: 0.1.3
- License: Apache-2.0
- Registry: PyPI (https://pypi.org/project/memwright/)

**MCP Registry:**
- Name: `io.github.bolnet/memwright`
- Version: 0.1.3
- Status: Active

**Build System:**
- Backend: hatchling
- Packages: `agent_memory` (wheel distribution)

**CI/CD:**
- GitHub Actions workflow: `.github/workflows/workflow.yml`
- Trigger: On release publication
- Process: Build → Publish to PyPI (OIDC trusted publisher in `environment: release`)

---

*Stack analysis: 2026-03-15*
