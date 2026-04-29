"""Synthetic Knowledge-Updates supersession suite.

Stress-tests Attestor's auto-supersession path
(``attestor/temporal/manager.py``) with hand-curated cases where:

1. Session 1 states a fact (e.g. "I live at 123 Main St").
2. Session 5 contradicts it (e.g. "I just moved to 456 Oak Ave").
3. Session 10 asks a question that must resolve to the NEW fact.

The 50 fixtures span 10 categories of contradiction (5 each):
numeric, categorical, temporal, preference, entity, locational,
intent, relational, count, status_binary.

**Metric:** percentage of cases where ``recall(question)`` returns the
newer fact above the older one. Target: 92-95% (per Gemini guidance).

Lives outside ``attestor.longmemeval`` because the data is
synthetic + repo-owned, not pulled from HuggingFace.
"""
