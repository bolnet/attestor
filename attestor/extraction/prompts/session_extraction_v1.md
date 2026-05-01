Extract ALL factual information from this conversation between {speaker_a} and {speaker_b}.

Return a JSON object with FOUR keys: "facts", "relations", "entities", "concepts".

## Facts (atomic statements)
For each fact, provide:
- content: A single, atomic, self-contained factual statement. Always include the person's name (never use "she", "he", or "they").
  Examples: "Caroline is single", "Melanie has two children", "Caroline's dog is named Max"
- entity: The primary person or thing this fact is about (e.g., "Caroline", "Melanie")
- category: One of: "personal", "career", "preference", "event", "plan", "location", "health", "general"
- tags: Relevant keywords for search (e.g., ["marital_status", "relationship", "single"])
- event_date: The date this fact refers to if mentioned or inferrable (ISO format YYYY-MM-DD), or null
- confidence: 0.0-1.0 how explicitly stated (1.0 = directly said, 0.7 = strongly implied)
- source_quote: Short verbatim quote (<=150 chars) from the conversation that supports this fact, or null
- kind: "list_item" when this fact is one element of an enumerated list from the source (shared source_quote across siblings); otherwise "atomic"

### List decomposition (CRITICAL)
If the source sentence enumerates N items for the SAME predicate
("I loved A, B, and C"; "She visited X, Y, Z"; "books I read: A, B, C"),
emit N SEPARATE facts — one per item — NOT a single compound fact.
The siblings share the same source_quote, event_date, entity, and tags,
and each has kind="list_item".

WRONG (one compound fact):
  {{"content": "Melanie read Charlotte's Web and Nothing is Impossible",
    "entity": "Melanie", "tags": ["books"], "kind": "atomic"}}

CORRECT (two atomic facts, same source_quote, kind=list_item):
  {{"content": "Melanie read Charlotte's Web",
    "entity": "Melanie", "tags": ["books", "reading"],
    "source_quote": "I loved Charlotte's Web and Nothing is Impossible",
    "kind": "list_item"}}
  {{"content": "Melanie read Nothing is Impossible",
    "entity": "Melanie", "tags": ["books", "reading"],
    "source_quote": "I loved Charlotte's Web and Nothing is Impossible",
    "kind": "list_item"}}

## Relations (entity-relationship triples)
For each relation:
- subject: The source entity (person, place, or thing)
- predicate: One of: knows, works_at, lives_in, has, owns, is, likes, dislikes, visited, studies_at, member_of, related_to, married_to, sibling_of, parent_of, child_of, friend_of, colleague_of, born_in, moved_to, traveled_to, started, ended, plans_to, wants_to
- object: The target entity
- event_date: ISO date (YYYY-MM-DD) if the relation has a temporal aspect, or null
- source_quote: Short verbatim quote from the conversation that supports this triple (<=150 chars), or null
- attributes: Optional JSON object of extra structured fields (e.g. {{"percentage": 0.9}}, {{"duration_years": 5}}), or null

Examples:
  {{"subject": "Caroline", "predicate": "works_at", "object": "Google", "event_date": "2024-01-15", "source_quote": "I started at Google last January", "attributes": null}}
  {{"subject": "Caroline", "predicate": "friend_of", "object": "Melanie", "event_date": null, "source_quote": "my friend Melanie and I", "attributes": null}}

### List decomposition for relations (CRITICAL)
When the source enumerates multiple objects for the same (subject, predicate),
emit ONE triple per object. Siblings share source_quote.

WRONG (collapsed object list):
  {{"subject": "Melanie", "predicate": "read",
    "object": "Charlotte's Web and Nothing is Impossible",
    "source_quote": "I loved Charlotte's Web and Nothing is Impossible"}}

CORRECT (one triple per book):
  {{"subject": "Melanie", "predicate": "read", "object": "Charlotte's Web",
    "source_quote": "I loved Charlotte's Web and Nothing is Impossible"}}
  {{"subject": "Melanie", "predicate": "read", "object": "Nothing is Impossible",
    "source_quote": "I loved Charlotte's Web and Nothing is Impossible"}}

## Entities (synthesized profile per person/thing)
For each distinct person, organization, or place mentioned, produce ONE profile aggregating everything known about them in this session:
- name: Canonical name (e.g. "Caroline", "Melanie", "TSMC")
- type: One of: "person", "organization", "place", "animal", "thing"
- profile: A paragraph (2-5 sentences) synthesizing who/what the entity is, based ONLY on this conversation. Include role, key traits, relationships, and recent activities. Always refer to the entity by name.
  Example: "Caroline is a transgender woman and LGBTQ activist working as a counselor. She is single, owns a dog named Max, and is close friends with Melanie. She is studying psychology and pursuing a counseling certification."
- tags: Keywords summarizing the entity (e.g. ["lgbtq", "counselor", "activist"])

## Concepts (synthesized themes / activity clusters / events)
For each recurring theme, activity cluster, or notable event in the conversation, produce ONE concept describing it:
- title: Short descriptive title (e.g. "Caroline's LGBTQ community participation", "Melanie's career transition to product management", "Joint trip to Barcelona 2023")
- description: A paragraph (2-5 sentences) synthesizing what this theme/activity/event is, who is involved, what happened, and when. Use names, not pronouns. Include specific dates, numbers, and places when stated.
  Example: "Caroline actively participates in the LGBTQ community through pride parades, a weekly support group she joined in March 2023, and an annual LGBTQ+ counseling conference. She attended pride events in May 2023 and serves as peer counselor in the support group."
- entities: List of entity names involved (e.g. ["Caroline", "Melanie"])
- tags: Keywords (e.g. ["lgbtq", "community", "activism"])
- event_date: ISO date for point-in-time events, or null for ongoing themes

Session date: {session_date}

Rules:
- Extract EVERY factual detail, no matter how small (names, dates, places, numbers, opinions, plans, activities)
- Each fact must be self-contained and readable without the conversation
- Include the person's name in every fact (never "she", "he", "they")
- Separate compound facts into individual atomic statements
- NEVER compound list items into one fact or triple — emit one fact AND one triple per list element (see "List decomposition" above)
- For temporal references like "next week" or "last month", resolve to dates relative to the session date
- Extract ALL relationships between people, places, organizations, and things mentioned
- Entity profiles and concepts must be SYNTHESIZED from the conversation, not copied verbatim
- A concept groups multiple related facts under one theme (e.g. "books Caroline is reading" rather than one concept per book) — concepts are a summary layer, they do NOT replace per-item facts and triples
- Do NOT include greetings, conversational filler, or meta-commentary

Conversation:
{conversation}

Return ONLY a valid JSON object with "facts", "relations", "entities", and "concepts" keys.