# mem-recall

Search memories using natural language. Uses multi-layer retrieval (tags, graph connections, semantic vectors) with score fusion to find the most relevant memories.

## Usage

```
/memwright:mem-recall <natural language query>
```

## Underlying Tool

`memory_recall` MCP tool with `query` parameter.

## Parameters

- **query** (required): Natural language search query
- **budget** (optional): Controls response size (number of results returned)

## Examples

- `/memwright:mem-recall What programming languages does the user prefer?`
- `/memwright:mem-recall Recent project decisions`
- `/memwright:mem-recall What do I know about deployment?`
- `/memwright:mem-recall Authentication patterns we discussed`

## Notes

- Returns ranked results with relevance scores
- Combines tag matching, graph traversal, and semantic vector search
- Use the `budget` parameter to control how many results are returned
- Best for open-ended queries where you want the most relevant memories
