"""Static prompt strings used by the LongMemEval pipeline.

Pulled verbatim from the legacy 2466-line ``attestor/longmemeval.py``
module — none of the wording has been edited as part of the package
split. Lives in its own file because the prompt bodies alone exceed
the 800-line per-module budget for ``runner.py``.
"""

from __future__ import annotations


DISTILL_PROMPT = (
    "You are a precise memory distillation agent for a long-term memory "
    "system. Extract durable facts from ONE conversation turn and return "
    "them as a JSON array of structured records.\n\n"
    "Each record must have these fields (all required, though lists may "
    "be empty):\n"
    '  "content":   third-person sentence (see rules below)\n'
    '  "speaker":   "user" or "assistant" (who made the underlying claim)\n'
    '  "claim_type": one of: fact, preference, recommendation, event, '
    "opinion, mentioned\n"
    '  "emphasis":  "explicit" (speaker named it specifically / endorsed '
    'it), "mentioned" (referenced in passing), or "implied"\n'
    '  "entities":  array of named entities the fact is about\n'
    '  "topics":    array of short topical tags (lowercase nouns)\n\n'
    "CLAIM_TYPE guide (pick the most specific match):\n"
    "  preference     — user expresses a like / dislike / constraint / "
    "priority (e.g. 'The user prefers dark chocolate').\n"
    "  recommendation — assistant explicitly suggests / recommends / "
    "endorses a NAMED target (e.g. 'The assistant recommended Roscioli').\n"
    "  event          — dated or schedulable occurrence (e.g. 'The user "
    "visited MoMA on 2023-06-15').\n"
    "  opinion        — speaker's subjective view without a concrete "
    "commitment.\n"
    "  mentioned      — a target was named but NOT explicitly recommended "
    "or endorsed (e.g. 'La Pergola was also discussed'). Use this when a\n"
    "  restaurant/book/place is referenced but the speaker did not make a\n"
    "  clear endorsement.\n"
    "  fact           — neutral factual statement that isn't any of the "
    "above (default fallback).\n\n"
    "EMPHASIS guide:\n"
    "  explicit  — speaker named it directly AND made it a focal point\n"
    "              (recommended X; user said they LOVE X; event pinpointed).\n"
    "  mentioned — named but not the focal point of the turn.\n"
    "  implied   — derivable from the turn but not literally stated.\n\n"
    "CONTENT RULES — the 'content' string itself:\n"
    "  1. PRESERVE literally: all proper nouns, dates, numbers, quantities, "
    "places, model names, prices, durations, colors, sizes, materials. "
    "Copy them verbatim from the turn.\n"
    "  2. REWRITE in third person from an outside observer's POV:\n"
    "       - User turn: 'The user ...', 'The user prefers ...'\n"
    "       - Assistant turn: 'The assistant told the user that ...', "
    "'The assistant recommended ...'\n"
    "     Never use 'I', 'you', 'we', 'my', 'your'.\n"
    "  3. RESOLVE every pronoun to its antecedent.\n"
    "  4. RESOLVE every relative time reference to an absolute date using "
    "the session_date as the anchor; write YYYY-MM-DD.\n"
    "     - For weekday-relative phrases ('last Friday', 'this Sunday', "
    "'next Monday'), step day-by-day from the anchor to the named weekday; "
    "do not default to ±7 days.\n"
    "     - For 'the WEEKDAY before/after DATE' (e.g. 'the Friday before "
    "July 15, 2023'), resolve relative to DATE, not to session_date: pick "
    "the first WEEKDAY strictly before/after DATE.\n"
    "     - For 'the week before X' / 'the weekend before X', resolve to "
    "the 7-day window (or Sat–Sun pair) immediately preceding X; if a "
    "single day is needed, prefer the matching weekday or X-7.\n"
    "     - When a relative phrase is ambiguous (no anchor in turn or "
    "session_date), KEEP the original phrase verbatim instead of guessing.\n"
    "  5. ONE FACT PER RECORD. If a turn has multiple distinct facts, "
    "emit multiple records.\n"
    "  6. NEVER FABRICATE. Only restate information literally in the turn.\n"
    "  7. Keep assistant-stated facts (recommendations, named entities, "
    "attributes, instructions, dated events). Skip ONLY pure pleasantries "
    "('thanks', 'ok'), generic puzzle answers with no user context, and "
    "off-topic trivia. When in doubt, KEEP.\n\n"
    "OUTPUT FORMAT:\n"
    "  - If the turn has ≥1 keep-worthy fact: emit a JSON array "
    "(starts with '[' and ends with ']'). No prose before or after. No "
    "code fences.\n"
    "  - If the turn is pure filler: emit exactly SKIP (nothing else).\n\n"
    "Worked examples:\n\n"
    "TURN: assistant: \"I'd recommend Roscioli for romantic Italian in "
    "Rome — it's a classic.\"\n"
    "OUTPUT:\n"
    "[\n"
    '  {"content": "The assistant recommended Roscioli for romantic Italian '
    'dinner in Rome.", "speaker": "assistant", "claim_type": '
    '"recommendation", "emphasis": "explicit", "entities": ["Roscioli", '
    '"Rome"], "topics": ["restaurant", "italian", "romantic", "dinner"]}\n'
    "]\n\n"
    "TURN: assistant: \"The Plesiosaur in the image has a blue scaly body "
    "and four flippers.\"\n"
    "OUTPUT:\n"
    "[\n"
    '  {"content": "The assistant told the user that the Plesiosaur in '
    'the image has a blue scaly body.", "speaker": "assistant", '
    '"claim_type": "fact", "emphasis": "explicit", "entities": '
    '["Plesiosaur"], "topics": ["animal", "color", "image"]},\n'
    '  {"content": "The assistant told the user that the Plesiosaur in '
    'the image has four flippers.", "speaker": "assistant", '
    '"claim_type": "fact", "emphasis": "explicit", "entities": '
    '["Plesiosaur"], "topics": ["animal", "anatomy", "image"]}\n'
    "]\n\n"
    "TURN: user: \"I prefer dark chocolate over milk chocolate and I "
    "can't stand cilantro.\"\n"
    "OUTPUT:\n"
    "[\n"
    '  {"content": "The user prefers dark chocolate over milk chocolate.", '
    '"speaker": "user", "claim_type": "preference", "emphasis": '
    '"explicit", "entities": ["dark chocolate", "milk chocolate"], '
    '"topics": ["food", "chocolate"]},\n'
    '  {"content": "The user cannot stand cilantro.", "speaker": "user", '
    '"claim_type": "preference", "emphasis": "explicit", "entities": '
    '["cilantro"], "topics": ["food", "dislike"]}\n'
    "]\n\n"
    "TURN: user: \"I visited the Rijksmuseum in Amsterdam on June 5, "
    "2023.\"\n"
    "OUTPUT:\n"
    "[\n"
    '  {"content": "The user visited the Rijksmuseum in Amsterdam on '
    '2023-06-05.", "speaker": "user", "claim_type": "event", "emphasis": '
    '"explicit", "entities": ["Rijksmuseum", "Amsterdam"], "topics": '
    '["museum", "travel"]}\n'
    "]\n\n"
    "TURN: assistant: \"Some options in Orlando for milkshakes include "
    "Toothsome Chocolate Emporium, but the Sugar Factory at Icon Park is "
    "the one famous for giant goblet shakes.\"\n"
    "OUTPUT:\n"
    "[\n"
    '  {"content": "The assistant identified the Sugar Factory at Icon '
    'Park as the Orlando spot famous for giant goblet milkshakes.", '
    '"speaker": "assistant", "claim_type": "recommendation", "emphasis": '
    '"explicit", "entities": ["Sugar Factory", "Icon Park", "Orlando"], '
    '"topics": ["dessert", "milkshake", "orlando"]},\n'
    '  {"content": "The assistant mentioned Toothsome Chocolate Emporium '
    'as another Orlando milkshake option.", "speaker": "assistant", '
    '"claim_type": "mentioned", "emphasis": "mentioned", "entities": '
    '["Toothsome Chocolate Emporium", "Orlando"], "topics": ["dessert", '
    '"orlando"]}\n'
    "]\n\n"
    "TURN: user: \"I had my charity 5K race last Sunday — beat my time.\"\n"
    "  (session_date = 2023-05-22, a Monday; last Sunday = 2023-05-21)\n"
    "OUTPUT:\n"
    "[\n"
    '  {"content": "The user ran a charity 5K race on 2023-05-21.", '
    '"speaker": "user", "claim_type": "event", "emphasis": "explicit", '
    '"entities": ["charity 5K race"], "topics": ["running", "charity"]}\n'
    "]\n\n"
    "TURN: assistant: \"Don’t forget your pottery workshop the Friday "
    "before your trip on July 15, 2023.\"\n"
    "  (Friday before 2023-07-15 (Sat) = 2023-07-14; resolve relative to "
    "the named DATE, not session_date)\n"
    "OUTPUT:\n"
    "[\n"
    '  {"content": "The assistant reminded the user about a pottery '
    'workshop on 2023-07-14, the Friday before the user’s trip on '
    '2023-07-15.", "speaker": "assistant", "claim_type": "event", '
    '"emphasis": "explicit", "entities": ["pottery workshop"], '
    '"topics": ["pottery", "workshop", "reminder"]}\n'
    "]\n\n"
    "TURN: assistant: \"Great, happy to help!\"\n"
    "OUTPUT:\n"
    "SKIP\n\n"
    "Turn context:\n"
    "  role = {role}\n"
    "  session_date = {session_date}\n\n"
    "Turn content:\n"
    "{content}\n\n"
    "Output: a JSON array of structured records, or SKIP."
)


ANSWER_PROMPT = (
    "You are answering a question based on an assistant's memory of a "
    "past chat history. The facts below are the user's memory — each is "
    "prefixed with [YYYY-MM-DD] for the session date when the turn was "
    "recorded.\n\n"
    "DECIDE THE QUESTION MODE FIRST (this governs everything else):\n"
    "  FACT mode — the question has a single correct answer the user "
    "or assistant previously stated. Hallmarks: when / where / who / "
    "what color / what (specific) / how many / how long / which "
    "(specific) / did I / have I / remind me what. Answer with the "
    "concrete value from the facts. If the facts have NO relevant "
    "information at all, respond exactly: I don't know.\n\n"
    "INFERENCE IS ALLOWED in FACT mode (RC2 fix — over-conservative\n"
    "refusal was a top failure mode):\n"
    "  - If dates exist for multiple events → you CAN compare them\n"
    "    for ordering (\"got router 01-15\" + \"set up thermostat 02-10\"\n"
    "    DOES support \"which device first?\" → router).\n"
    "  - \"Got/purchased/received a device\" implies setup/use occurred.\n"
    "  - \"Took N flights with airline X\" implies flew with X N times.\n"
    "  - Numeric facts combine: \"living in US 5 years\" + \"32 years\n"
    "    old\" → \"how old when moved to US?\" = 27.\n"
    "  - Match weekend/weekday names to dates (03-15 = Saturday).\n"
    "  - ONLY say \"I don't know\" when facts have NO relevant info\n"
    "    — not when they lack a literal explicit answer phrase.\n\n"
    "DATE DISAMBIGUATION (RC3 fix):\n"
    "  - [YYYY-MM-DD] prefix = CONVERSATION date (when recorded).\n"
    "  - Dates inside the fact TEXT = EVENT dates (when things\n"
    "    actually happened). Always prefer EVENT dates for arithmetic.\n"
    "    e.g. [2023-02-01] \"took guitar to tech on 2023-02-25\" →\n"
    "    use 02-25 not 02-01.\n"
    "  - \"How many weeks ago\" → round to NEAREST week (22 days = 3).\n\n"
    "TEMPORAL ANCHORING (RC5 fix):\n"
    "  - \"N weeks/months ago\" → compute target = question_date − N,\n"
    "    then pick the event whose date is CLOSEST to target. Not\n"
    "    the most keyword-similar one.\n"
    "  - Multiple instances of same event type (e.g. two museum\n"
    "    visits): pick the one closest to the implied date.\n\n"
    "LIST COMPLETENESS (RC6 fix):\n"
    "  - If asked for an ordered list, you MUST list ALL items the\n"
    "    facts support. Stopping after one item counts as wrong.\n\n"
    "  RECOMMENDATION mode — the question asks for tips, advice, "
    "ideas, suggestions, or a tailored proposal. Hallmarks: recommend / "
    "suggest / propose / advise / help me pick / any tips / any advice / "
    "any ideas / what should I / what would you / I'm looking for / I "
    "need help with. For these you MUST NOT respond \"I don't know\" — "
    "use the user's stored preferences, habits, constraints, and past "
    "experiences to produce a concrete, user-specific proposal. You may "
    "blend stored facts with sensible domain knowledge, but the "
    "response must be obviously tailored to THIS user (cite relevant "
    "stored facts in parentheses). Generic boilerplate is WRONG.\n\n"
    "EDGE CASES:\n"
    "  - \"Can you remind me what color …\" → FACT (recall)\n"
    "  - \"What's my favorite …\" → FACT (recall of a stated preference)\n"
    "  - \"Can you recommend a hotel for …\" → RECOMMENDATION\n"
    "  - \"Any tips for keeping my kitchen clean?\" → RECOMMENDATION\n\n"
    "DATE ARITHMETIC — applies to FACT mode questions about durations, "
    "gaps, or event dates:\n"
    "  1. Identify every date relevant to the question (event dates, "
    "the question date, reference dates in the question itself).\n"
    "  2. \"how many days/weeks/months/years between X and Y\" → "
    "     compute |date_Y − date_X| in the requested unit. Measure\n"
    "     BETWEEN the two events named in the question, not \"from now\".\n"
    "  3. \"how many days/months ago\" → anchor is the question_date, "
    "     not today.\n"
    "  4. \"when I did X\", \"when I made the cake\" — that clause IS\n"
    "     the anchor; do not substitute the question_date.\n"
    "  5. Months: full calendar months. 2022-10-22 → 2023-03-22 = 5.\n"
    "  6. Days: calendar count; verify month-by-month.\n\n"
    "Before your final answer, think step-by-step inside "
    "<reasoning>...</reasoning>. Include the mode decision (FACT vs "
    "RECOMMENDATION), the relevant facts you'll cite, and any "
    "arithmetic. Then output the final answer AFTER </reasoning> on "
    "its own — concise and concrete.\n\n"
    "WORKED EXAMPLES (showing both modes):\n\n"
    "Example A — FACT, temporal:\n"
    "  Question (asked on 2023-07-20): How many days between my visit\n"
    "  to the museum and the concert?\n"
    "  Facts:\n"
    "    - [2023-06-05] The user visited the Rijksmuseum in Amsterdam.\n"
    "    - [2023-06-17] The user attended a jazz concert at Bimhuis.\n"
    "  <reasoning>\n"
    "  Mode: FACT (how many days between X and Y).\n"
    "  Dates: 2023-06-05 (museum) and 2023-06-17 (concert).\n"
    "  Diff: 2023-06-17 − 2023-06-05 = 12 days.\n"
    "  </reasoning>\n"
    "  12 days.\n\n"
    "Example B — FACT, recall (single-session-assistant):\n"
    "  Question: What color was the Plesiosaur the assistant described?\n"
    "  Facts:\n"
    "    - [2023-04-10] The assistant told the user the Plesiosaur in\n"
    "      the image has a blue scaly body and four flippers.\n"
    "  <reasoning>\n"
    "  Mode: FACT (what color — one correct value).\n"
    "  Fact [2023-04-10] states the Plesiosaur had a blue scaly body.\n"
    "  </reasoning>\n"
    "  Blue.\n\n"
    "Example C — RECOMMENDATION, hotel:\n"
    "  Question (asked on 2024-03-10): Can you suggest a hotel for my\n"
    "  trip to Lisbon?\n"
    "  Facts:\n"
    "    - [2022-04-12] The user stayed at Casa das Janelas com Vista\n"
    "      in Lisbon and enjoyed it.\n"
    "    - [2023-09-03] The user said they prefer boutique hotels over\n"
    "      chains and love rooftop views of the water.\n"
    "  <reasoning>\n"
    "  Mode: RECOMMENDATION. User preferences: boutique + rooftop water\n"
    "  views. Past win: Casa das Janelas com Vista. Combine with\n"
    "  domain knowledge to propose concrete boutique options with water\n"
    "  views.\n"
    "  </reasoning>\n"
    "  Based on your preference for boutique hotels with rooftop water\n"
    "  views (per 2023-09-03):\n"
    "  - Memmo Alfama — boutique, rooftop plunge pool over the Tagus.\n"
    "  - Santiago de Alfama — boutique, Alfama rooftop, 32 rooms.\n"
    "  - Casa das Janelas com Vista — you already enjoyed this in\n"
    "    2022-04-12 and it still fits your stated preferences.\n\n"
    "Example D — RECOMMENDATION, kitchen tips:\n"
    "  Question: Any tips for keeping my kitchen organized?\n"
    "  Facts:\n"
    "    - [2024-01-14] The user bought a magnetic knife strip.\n"
    "    - [2024-02-02] The user said they dislike countertop clutter.\n"
    "    - [2023-11-10] The user has granite countertops.\n"
    "  <reasoning>\n"
    "  Mode: RECOMMENDATION. Build on existing tools (magnetic knife\n"
    "  strip), respect granite (no vinegar/lemon), target declutter.\n"
    "  </reasoning>\n"
    "  Build on what you already have:\n"
    "  - Move cutting boards vertical beside the magnetic knife strip\n"
    "    you installed (per 2024-01-14) to keep counters clear.\n"
    "  - Use pH-neutral cleaner on granite; avoid vinegar or lemon.\n"
    "  - Add a two-tier pull-out under the sink to corral bottles —\n"
    "    directly addresses your dislike of countertop clutter\n"
    "    (per 2024-02-02).\n\n"
    "Now answer the real question.\n\n"
    "Question (asked on {question_date}):\n{question}\n\n"
    "Facts:\n{context}\n\n"
    "Output:\n"
    "<reasoning>...your mode decision + cited facts + arithmetic...</reasoning>\n"
    "<final answer only>"
)


PERSONALIZATION_PROMPT = ANSWER_PROMPT  # unified into ANSWER_PROMPT


VERIFY_PROMPT = (
    "Double-check the AI's answer below against the facts. You are a "
    "second-pass verifier. Your job is to catch date-arithmetic mistakes, "
    "misread temporal anchors, and questions where the AI abstained "
    "despite sufficient evidence.\n\n"
    "Question (asked on {question_date}):\n{question}\n\n"
    "Facts:\n{context}\n\n"
    "AI's first answer:\n{first_answer}\n\n"
    "Rules:\n"
    "  1. If the first answer is correct, repeat it verbatim (same number, "
    "same date, same phrase). Do NOT rephrase or elaborate.\n"
    "  2. If the first answer has an arithmetic error, recompute and "
    "output the corrected final answer.\n"
    "  3. If the first answer abstained (\"I don't know\") but the facts "
    "contain a specific answer, replace the abstention with the correct "
    "answer.\n"
    "  4. If the facts truly do not support an answer, keep \"I don't know\".\n\n"
    "Output ONLY the final answer on one line. No reasoning, no prefix, "
    "no explanation."
)


JUDGE_PROMPT = (
    "You are judging whether an AI's answer is CORRECT or WRONG given the "
    "gold answer. Be generous — if the AI's answer semantically matches or "
    "contains the gold answer, mark CORRECT. For dates, accept equivalent "
    "formats.\n\n"
    "Category-specific rubric (question category: {category}):\n"
    "  - temporal-reasoning: the answer must include the correct date or "
    "period; if the AI paraphrases a relative phrase (\"last year\") instead "
    "of resolving it, mark WRONG.\n"
    "  - knowledge-update: the answer must reflect the LATEST state, not an "
    "older superseded value; WRONG if stale.\n"
    "  - abstention: if the gold answer is an abstention (e.g. \"I don't "
    "know\", \"not mentioned\"), accept any reasonable abstention; hallucinated "
    "facts are WRONG.\n"
    "  - Other categories: match gold answer on substance, not wording.\n\n"
    "Question: {question}\n"
    "Gold answer: {expected}\n"
    "AI answer: {generated}\n\n"
    "Return JSON with keys \"reasoning\" (one sentence) and \"label\" "
    "(CORRECT or WRONG)."
)


PERSONALIZATION_JUDGE_PROMPT = (
    "You are scoring the PERSONALIZATION QUALITY of an AI's answer to a "
    "recommendation question. The user asked for advice / tips / "
    "suggestions, and the AI gave a tailored answer. Your job is to "
    "decide whether the answer is sufficiently personalized to THIS "
    "user's stored memory.\n\n"
    "Inputs:\n"
    "  Question: {question}\n"
    "  Stored user facts available to the AI:\n{context}\n"
    "  AI answer: {generated}\n"
    "  Reference 'preferred response' specification: {expected}\n\n"
    "Score CRITERIA — both must be true for label CORRECT:\n"
    "  (a) The AI cites or references at least one stored user fact "
    "      (preference, habit, past experience, constraint, named entity)\n"
    "      that is relevant to the question.\n"
    "  (b) The AI's recommendations are concrete (named items, specific "
    "      methods) and align with the reference specification's intent. "
    "      Generic boilerplate ('focus on a few key habits') without "
    "      user-specific tailoring is WRONG.\n\n"
    "Edge cases:\n"
    "  - If the AI abstains ('I don't know') for a recommendation "
    "    question, that is WRONG.\n"
    "  - If the stored facts don't support strong tailoring AND the AI "
    "    politely admits limited context while still giving a useful "
    "    starting list, that is CORRECT.\n"
    "  - Citing a stored fact that contradicts the question's intent is "
    "    WRONG.\n\n"
    "Return JSON with keys \"reasoning\" (one sentence) and \"label\" "
    "(CORRECT or WRONG)."
)

