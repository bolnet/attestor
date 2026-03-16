# mem-health

Check health of all memory system components (SQLite, ChromaDB, NetworkX, Retrieval Pipeline). Call this first when troubleshooting memory issues.

## Usage

```
/memwright:mem-health
```

## Underlying Tool

`memory_health` MCP tool (no parameters required).

## Parameters

None.

## Examples

- `/memwright:mem-health` -- shows status of all components

## Notes

- Call this first when troubleshooting any memory issues
- Shows status, memory counts, and latency for each component
- Components checked: SQLite store, ChromaDB vector store, NetworkX graph, Retrieval pipeline
- A healthy system shows all components as operational
