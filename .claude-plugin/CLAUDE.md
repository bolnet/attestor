# Memwright - Automatic Memory

Memwright provides persistent memory across sessions. Memories are automatically captured and recalled.

## Available MCP Tools

- `memory_recall` - Find relevant memories (primary tool, uses multi-layer search)
- `memory_add` - Store a new fact or decision
- `memory_search` - Search with filters (category, entity, date range)
- `memory_timeline` - Get chronological history for an entity
- `memory_health` - Check system status (call first to verify)
- `memory_stats` - Get store statistics
- `memory_get` - Retrieve a specific memory by ID
- `memory_forget` - Archive a memory

## Usage Guidelines

- Call `memory_health` at session start to verify the system is working
- Use `memory_recall` as the primary search -- it fuses tag, graph, and vector results
- When the user shares preferences, decisions, or important context, store with `memory_add`
- Use specific tags and categories for better retrieval:
  - Categories: career, project, preference, personal, technical, general
  - Tags: descriptive keywords like ["python", "architecture", "preference"]
- Set `entity` field for facts about specific things (tools, companies, people)
- Use `memory_timeline` to see how facts about an entity evolved over time
