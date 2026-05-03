"""PII detection + opt-in redaction at ingest.

Built for enterprise / regulated-industry buyers (healthcare, fintech,
legal) that need PII handling at the memory-write layer rather than the
application layer. The detector ships with:

  * A regex baseline (fast — well under 5ms typical, <50ms worst case).
  * Luhn-checked credit-card detection (avoids 16-digit timestamp + ID
    false positives).
  * Context-keyword gating for DOB / MRN (avoid bare-date false positives).
  * An optional LLM-judged path for harder cases (names, addresses)
    that the regex layer deliberately skips.

The detector is invoked from ``AgentMemory.add()`` when
``stack.compliance.pii.mode`` is anything other than ``off``. Three
modes are wired:

    off      — detector never runs. Byte-identical to the legacy add().
    flag     — detect, write findings to ``metadata['_pii_findings']``,
               leave content unchanged.
    redact   — detect AND replace each finding with a typed redaction
               token (``[REDACTED:EMAIL]``, ``[REDACTED:SSN]``, etc.);
               metadata flags ``_pii_original_sealed=True`` so callers
               know the content has been transformed.

Failure isolation: a detector exception is caught by the caller
(``AgentMemory.add()``) and turned into a ``_pii_detector_error=True``
metadata flag — ingest never blocks on PII tooling failure. The doc
store path is the only hard dependency.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("attestor.compliance.pii")


# ─── Public dataclasses ─────────────────────────────────────────────────


@dataclass(frozen=True)
class PIIFinding:
    """One PII match.

    ``type`` is one of: email | phone | ssn | credit_card | ip |
    url_with_auth | name | address | dob | mrn | other.
    ``span`` is ``(start_char, end_char)`` (Python-style half-open
    interval over the original content string).
    ``detector`` distinguishes ``regex_v1`` vs ``llm_v1`` so audit
    consumers can tell the rule-based and LLM-judged findings apart.
    """

    type: str
    span: tuple[int, int]
    confidence: float
    detector: str


@dataclass(frozen=True)
class PIIResult:
    """Aggregated detection output.

    ``redacted_content`` is ``None`` unless the caller asked for
    ``mode='redact'``. ``elapsed_ms`` is the wall-clock cost of the
    detection pass — useful as a smoke metric when running on large
    ingest streams.
    """

    findings: tuple[PIIFinding, ...]
    redacted_content: str | None
    elapsed_ms: float


@dataclass(frozen=True)
class PIIConfig:
    """Runtime mirror of ``stack.compliance.pii``.

    The :class:`attestor.config.PIICfg` dataclass carries the same
    fields and is what the YAML loader builds. Both shapes are kept so
    callers can construct a config in-process without going through
    the loader.
    """

    mode: str = "off"
    llm_model: str | None = None


# Recognised modes — surfaced for the YAML loader's validation hook.
VALID_MODES: tuple[str, ...] = ("off", "flag", "redact", "llm")


# ─── Redaction tokens ───────────────────────────────────────────────────

_REDACTION_TOKENS: dict[str, str] = {
    "email": "[REDACTED:EMAIL]",
    "phone": "[REDACTED:PHONE]",
    "ssn": "[REDACTED:SSN]",
    "credit_card": "[REDACTED:CREDIT_CARD]",
    "ip": "[REDACTED:IP]",
    "url_with_auth": "[REDACTED:URL_WITH_AUTH]",
    "name": "[REDACTED:NAME]",
    "address": "[REDACTED:ADDRESS]",
    "dob": "[REDACTED:DOB]",
    "mrn": "[REDACTED:MRN]",
    "other": "[REDACTED:OTHER]",
}


def _token_for(pii_type: str) -> str:
    """Resolve the redaction token for a PII type — falls back to a
    generic ``[REDACTED:OTHER]`` for forward-compat with new types."""
    return _REDACTION_TOKENS.get(pii_type, _REDACTION_TOKENS["other"])


# ─── Regex catalog ──────────────────────────────────────────────────────
#
# Each entry is a (compiled_regex, pii_type, confidence) tuple. We
# deliberately compile once at module import to keep the per-call cost
# under a few microseconds. Order matters — more-specific patterns
# (URL with embedded auth) run BEFORE less-specific ones (raw email,
# IP) so we don't double-flag a single span.

_EMAIL_RE = re.compile(
    r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b",
)

# Phone: optional country code, optional separator, 3-3-4 grouping.
# Use a positive look-ahead boundary so '2024-01-15' (an ISO date)
# doesn't match — the area code "2024" is a year, not a 3-digit area
# code, and the look-around forces the next char after the digit run
# to NOT be another digit (which knocks ISO-8601 + Unix timestamps out).
_PHONE_RE = re.compile(
    r"(?<!\d)"
    r"(?:\+?\d{1,3}[\s.\-]?)?"
    r"\(?\d{3}\)?[\s.\-]\d{3}[\s.\-]\d{4}"
    r"(?!\d)",
)

# SSN: dashed only. Pure-9-digit unspaced is too ambiguous (matches
# many IDs / phone numbers without separators), so we keep it tight to
# the dashed form which is unambiguously American SSN-shaped.
_SSN_RE = re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)")

# Credit card: 13-19 digits, optionally separated by spaces / dashes.
# The Luhn check below is the actual gate — the regex only proposes
# candidates. Boundary look-arounds prevent matching inside longer
# digit runs (e.g. trace ids, sha sums).
_CC_RE = re.compile(
    r"(?<!\d)"
    r"(?:\d[ \-]?){13,19}"
    r"(?!\d)",
)

# URL with embedded basic-auth credentials. Match BEFORE the bare
# email regex so we don't double-claim the user@host portion.
_URL_AUTH_RE = re.compile(
    r"https?://[^\s:/@]+:[^\s@]+@[^\s]+",
)

# IP with embedded auth. Tighter than IP-only since the 'auth@'
# anchor avoids most date / version-number false positives.
_IP_AUTH_RE = re.compile(
    r"\b\w+:\w+@(?:\d{1,3}\.){3}\d{1,3}\b",
)

# DOB: m/d/yyyy or m/d/yy. Gated on a context keyword (dob | birth |
# born | date of birth | d.o.b) appearing within ``_DOB_CONTEXT_WINDOW``
# characters BEFORE the match — this is the false-positive guard the
# spec requires.
_DOB_RE = re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b")
_DOB_CONTEXT_RE = re.compile(
    r"\b(?:dob|d\.o\.b|date\s+of\s+birth|birth\s*date|born|birthday)\b",
    re.IGNORECASE,
)
_DOB_CONTEXT_WINDOW = 40

# MRN: 6+ digits with a "MRN" / "Medical Record" gate. Like DOB, we
# refuse to flag bare digit runs as MRN — too prone to false positives
# on order ids, trace ids, etc.
_MRN_RE = re.compile(r"\b\d{6,12}\b")
_MRN_CONTEXT_RE = re.compile(
    r"\b(?:mrn|medical\s+record(?:\s+number)?)\b",
    re.IGNORECASE,
)
_MRN_CONTEXT_WINDOW = 60


# ─── Luhn check ─────────────────────────────────────────────────────────


def _luhn_valid(digits: str) -> bool:
    """Return True iff ``digits`` (a string of ASCII digits) passes the
    Luhn checksum. False on empty input or on any non-digit char."""
    if not digits or not digits.isdigit():
        return False
    total = 0
    # Iterate right-to-left, doubling every second digit and folding
    # two-digit results back into a single digit (Luhn classic).
    for i, ch in enumerate(reversed(digits)):
        n = ord(ch) - ord("0")
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


# ─── DOB / MRN context gating ───────────────────────────────────────────


def _has_context(
    content: str, span: tuple[int, int], pattern: re.Pattern[str], window: int,
) -> bool:
    """True iff ``pattern`` matches anywhere in the ``window`` characters
    before ``span[0]`` OR within the same line after ``span[1]`` (so
    'Patient DOB: 1/15/1990' AND '1/15/1990 (DOB)' both qualify)."""
    pre_start = max(0, span[0] - window)
    before = content[pre_start:span[0]]
    after = content[span[1]:span[1] + window]
    return bool(pattern.search(before) or pattern.search(after))


# ─── Detection internals ────────────────────────────────────────────────


def _detect_regex(content: str) -> list[PIIFinding]:
    """Run every regex lane and return findings sorted by span start.

    Overlap rule: if two findings cover the same span, the higher-
    confidence one wins. This matters when a URL with auth would
    otherwise also be reported as an email (the 'user@host.com'
    fragment).
    """
    raw: list[PIIFinding] = []

    # URL with auth → checked first so it claims its span before the
    # email lane sees it.
    for m in _URL_AUTH_RE.finditer(content):
        raw.append(PIIFinding(
            type="url_with_auth",
            span=m.span(),
            confidence=0.99,
            detector="regex_v1",
        ))

    # IP with auth.
    for m in _IP_AUTH_RE.finditer(content):
        raw.append(PIIFinding(
            type="ip",
            span=m.span(),
            confidence=0.95,
            detector="regex_v1",
        ))

    for m in _EMAIL_RE.finditer(content):
        raw.append(PIIFinding(
            type="email",
            span=m.span(),
            confidence=0.99,
            detector="regex_v1",
        ))

    for m in _PHONE_RE.finditer(content):
        raw.append(PIIFinding(
            type="phone",
            span=m.span(),
            confidence=0.92,
            detector="regex_v1",
        ))

    for m in _SSN_RE.finditer(content):
        raw.append(PIIFinding(
            type="ssn",
            span=m.span(),
            confidence=0.98,
            detector="regex_v1",
        ))

    for m in _CC_RE.finditer(content):
        candidate = m.group(0)
        digits = re.sub(r"[ \-]", "", candidate)
        if 13 <= len(digits) <= 19 and _luhn_valid(digits):
            raw.append(PIIFinding(
                type="credit_card",
                span=m.span(),
                confidence=0.95,
                detector="regex_v1",
            ))

    # DOB — context-gated.
    for m in _DOB_RE.finditer(content):
        if _has_context(content, m.span(), _DOB_CONTEXT_RE, _DOB_CONTEXT_WINDOW):
            raw.append(PIIFinding(
                type="dob",
                span=m.span(),
                confidence=0.85,
                detector="regex_v1",
            ))

    # MRN — context-gated.
    for m in _MRN_RE.finditer(content):
        if _has_context(content, m.span(), _MRN_CONTEXT_RE, _MRN_CONTEXT_WINDOW):
            raw.append(PIIFinding(
                type="mrn",
                span=m.span(),
                confidence=0.80,
                detector="regex_v1",
            ))

    return _dedupe_overlaps(raw)


def _dedupe_overlaps(findings: list[PIIFinding]) -> list[PIIFinding]:
    """Drop findings whose span is fully contained within another
    higher-confidence finding's span. This is the rule that prevents
    'jane@example.com' inside 'https://jane:pw@example.com' from
    showing up as both 'email' AND 'url_with_auth'."""
    if not findings:
        return findings
    # Sort by span start, then by descending confidence (so the higher-
    # confidence overlap is processed first).
    sorted_findings = sorted(
        findings, key=lambda f: (f.span[0], -f.confidence),
    )
    kept: list[PIIFinding] = []
    for f in sorted_findings:
        # Drop ``f`` if any ``k`` already kept fully contains its span.
        if any(
            k.span[0] <= f.span[0] and k.span[1] >= f.span[1] and k is not f
            for k in kept
        ):
            continue
        kept.append(f)
    # Return sorted-by-span — caller code (redaction in particular)
    # depends on this ordering.
    return sorted(kept, key=lambda f: f.span[0])


# ─── LLM-judged lane (opt-in) ───────────────────────────────────────────


_LLM_PROMPT = """You are a PII detector. Look at the content below and \
return ONLY a JSON list of findings. Each finding must be an object with:
  - "type": one of {types}
  - "span": [start_char, end_char] using zero-based Python char indexing
  - "confidence": float in [0.0, 1.0]

Focus on what regex CANNOT find: person names, street addresses, and \
free-form references that imply identity. Do NOT report email / phone \
/ SSN / credit-card — those are handled by a regex layer upstream.

If you find no PII, return an empty list: []

CONTENT:
{content}

JSON:"""


def _detect_llm(
    content: str,
    *,
    llm_model: str,
    llm_client: Any,
) -> list[PIIFinding]:
    """Optional LLM-judged path. Returns extra findings (names, addresses)
    that complement the regex baseline. Failure-isolated: a malformed
    response or LLM error logs a warning and returns an empty list — the
    regex findings continue to count."""
    import json

    types = ", ".join(sorted({"name", "address", "dob", "mrn", "other"}))
    prompt = _LLM_PROMPT.format(types=types, content=content)
    try:
        response = llm_client.chat.completions.create(
            model=llm_model,
            max_tokens=300,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        text = (response.choices[0].message.content or "").strip()
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "PII LLM lane failed (%s: %s); using regex-only findings",
            type(e).__name__, e,
        )
        return []

    # Strip markdown fences if the model wrapped the JSON.
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()

    try:
        parsed = json.loads(text)
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "PII LLM lane returned non-JSON (%s: %s); response=%r",
            type(e).__name__, e, text[:120],
        )
        return []

    if not isinstance(parsed, list):
        return []

    out: list[PIIFinding] = []
    for entry in parsed:
        if not isinstance(entry, dict):
            continue
        try:
            t = str(entry["type"])
            span = entry["span"]
            start, end = int(span[0]), int(span[1])
            conf = float(entry.get("confidence", 0.7))
        except (KeyError, ValueError, TypeError):
            continue
        if start < 0 or end > len(content) or start >= end:
            continue
        out.append(PIIFinding(
            type=t, span=(start, end), confidence=conf, detector="llm_v1",
        ))
    return out


# ─── Redaction ──────────────────────────────────────────────────────────


def redact_content(content: str, findings: list[PIIFinding]) -> str:
    """Return ``content`` with every finding's span replaced by a typed
    redaction token. The replacement happens right-to-left so the
    spans on remaining findings stay valid through the rewrite.

    Public helper — separable from :func:`detect_pii` so callers can
    pre-compute findings (or stitch findings from multiple sources)
    and then redact in one pass.
    """
    if not findings:
        return content
    sorted_findings = sorted(findings, key=lambda f: f.span[0], reverse=True)
    out = content
    for f in sorted_findings:
        token = _token_for(f.type)
        out = out[:f.span[0]] + token + out[f.span[1]:]
    return out


# ─── Public entrypoint ──────────────────────────────────────────────────


def detect_pii(
    content: str,
    *,
    mode: str = "flag",
    llm_model: str | None = None,
    llm_client: Any | None = None,
) -> PIIResult:
    """Detect PII in ``content``. Optionally redact when ``mode='redact'``.

    Modes:
      * ``flag``   — return findings; ``redacted_content=None``.
      * ``redact`` — return findings AND ``redacted_content`` (rewritten
                     with typed tokens).
      * ``llm``    — run the regex baseline AND the optional LLM lane;
                     return findings; ``redacted_content=None`` (caller
                     redacts separately if desired).

    For ``mode='off'`` the caller should not invoke this function at
    all; calling it explicitly with ``mode='off'`` returns an empty
    result.
    """
    if mode not in VALID_MODES:
        raise ValueError(
            f"unknown PII mode {mode!r}; expected one of {list(VALID_MODES)}"
        )

    t0 = time.monotonic()

    if mode == "off" or not content:
        return PIIResult(
            findings=(),
            redacted_content=None,
            elapsed_ms=round((time.monotonic() - t0) * 1000, 3),
        )

    findings = _detect_regex(content)

    if mode == "llm":
        if llm_model and llm_client is not None:
            extras = _detect_llm(
                content, llm_model=llm_model, llm_client=llm_client,
            )
            # Avoid duplicating spans the regex already covers.
            existing_spans = {f.span for f in findings}
            findings.extend(
                f for f in extras if f.span not in existing_spans
            )
            findings = _dedupe_overlaps(findings)
        else:
            logger.debug(
                "PII mode=llm but llm_model/client missing; "
                "falling back to regex-only findings"
            )

    redacted: str | None = None
    if mode == "redact" and findings:
        redacted = redact_content(content, list(findings))

    elapsed_ms = round((time.monotonic() - t0) * 1000, 3)
    return PIIResult(
        findings=tuple(findings),
        redacted_content=redacted,
        elapsed_ms=elapsed_ms,
    )
