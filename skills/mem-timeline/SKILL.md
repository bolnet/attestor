# mem-timeline

View chronological history of memories about a specific entity (person, project, tool, company). Shows how knowledge about that entity evolved over time.

## Usage

```
/memwright:mem-timeline <entity name>
```

## Underlying Tool

`memory_timeline` MCP tool with `entity` parameter.

## Parameters

- **entity** (required): Name of the entity to get timeline for (e.g., "Python", "deployment", "auth-service")

## Examples

- `/memwright:mem-timeline Python` -- shows all memories about Python in date order
- `/memwright:mem-timeline auth-service` -- tracks how auth service decisions evolved
- `/memwright:mem-timeline React` -- see when and what was discussed about React

## Notes

- Shows both active and superseded memories
- Useful for tracking how knowledge evolved over time
- Memories are displayed in chronological order
- Superseded memories are marked so you can see what changed
