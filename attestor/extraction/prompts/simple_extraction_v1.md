Extract atomic facts from this conversation that would be useful to remember across sessions.

For each fact, provide:
- content: A single, self-contained factual statement
- tags: List of relevant tags
- category: One of "career", "project", "preference", "personal", "technical", "general"
- entity: The primary entity this fact is about (company, person, tool, etc.), or null

Return a JSON array of objects. Only include facts that are clearly stated, not speculative.

Conversation:
{conversation}

Return ONLY valid JSON array, no other text.