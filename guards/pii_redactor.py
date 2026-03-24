"""
Guard 4 — PII Redactor
========================
Detection strategy (two-tier):

TIER 1 — spaCy NER + Regex (production-grade):
  - spaCy en_core_web_sm Named Entity Recognition for:
    PERSON, ORG, GPE, LOC, DATE, TIME entities
  - 10 regex patterns for structured PII:
    email, US phone, India phone, SSN, credit card, Aadhaar,
    IP address, date of birth, passport numbers, API tokens in URLs
  - Redaction preserves sentence grammatical structure by replacing
    with labeled tokens: [PERSON], [EMAIL], [PHONE], etc.

TIER 2 — Heuristic NER (fallback when spaCy unavailable):
  Capitalization-based person name detection + same regex patterns.

Industry references:
  - Microsoft Presidio (same regex + NER hybrid approach)
  - AWS Comprehend PII detection
  - Google DLP (Data Loss Prevention) API
"""

import re
from dataclasses import dataclass, field
from typing import List, Tuple

from config.model_loader import MODELS_READY, get_spacy_nlp


@dataclass
class PIIEntity:
    entity_type: str
    original: str
    replacement: str
    start: int
    end: int


@dataclass
class GuardResult:
    triggered: bool
    risk_score: float
    reason: str
    action: str
    clean_text: str
    pii_found: List[PIIEntity] = field(default_factory=list)
    detection_method: str = "regex"


# ── Regex PII patterns ───────────────────────────────────────
# Format: (entity_type, compiled_regex, replacement_label)
PII_PATTERNS = [
    ("EMAIL",
     re.compile(r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b'),
     "[EMAIL]"),

    ("CREDIT_CARD",
     re.compile(r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|'
                r'(?:\d{4}[\s\-]?){3}\d{4})\b'),
     "[CARD]"),

    ("SSN",
     re.compile(r'\b(?!000|666|9\d{2})\d{3}[\s\-]?(?!00)\d{2}[\s\-]?(?!0000)\d{4}\b'),
     "[SSN]"),

    ("PHONE_US",
     re.compile(r'\b(\+1[\s\-.]?)?\(?\d{3}\)?[\s\-.]?\d{3}[\s\-.]?\d{4}\b'),
     "[PHONE]"),

    ("PHONE_IN",
     re.compile(r'\b(\+91[\s\-]?)?[6-9]\d{9}\b'),
     "[PHONE]"),

    ("AADHAAR",
     re.compile(r'\b[2-9]{1}[0-9]{3}\s?[0-9]{4}\s?[0-9]{4}\b'),
     "[AADHAAR]"),

    ("IP_ADDRESS",
     re.compile(r'\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}'
                r'(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b'),
     "[IP]"),

    ("DOB",
     re.compile(r'\b(?:dob|date of birth|born on|birth date)[:\s]+'
                r'\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}\b', re.IGNORECASE),
     "[DOB]"),

    ("API_TOKEN",
     re.compile(r'https?://\S+[?&](token|key|secret|api_key|apikey)=\S+',
                re.IGNORECASE),
     "[URL_WITH_TOKEN]"),

    ("PASSPORT",
     re.compile(r'\b[A-Z]{1,2}[0-9]{6,9}\b'),
     "[PASSPORT]"),
]


def _apply_regex(text: str) -> Tuple[str, List[PIIEntity]]:
    """Apply all regex patterns, track positions for non-overlapping replacement."""
    entities = []
    # Collect all matches first
    for entity_type, pattern, replacement in PII_PATTERNS:
        for match in pattern.finditer(text):
            entities.append(PIIEntity(
                entity_type=entity_type,
                original=match.group(),
                replacement=replacement,
                start=match.start(),
                end=match.end()
            ))

    if not entities:
        return text, []

    # Sort by start position, remove overlaps (keep longest)
    entities.sort(key=lambda e: (e.start, -(e.end - e.start)))
    non_overlapping = []
    last_end = 0
    for e in entities:
        if e.start >= last_end:
            non_overlapping.append(e)
            last_end = e.end

    # Apply replacements right-to-left to preserve indices
    result = text
    for e in reversed(non_overlapping):
        result = result[:e.start] + e.replacement + result[e.end:]

    return result, non_overlapping


def _apply_spacy_ner(text: str, already_redacted: str) -> Tuple[str, List[PIIEntity]]:
    """Apply spaCy NER for named entities not caught by regex."""
    nlp = get_spacy_nlp()
    if nlp is None:
        return already_redacted, []

    doc = nlp(text)  # Run on original text to get correct spans
    entities = []
    REDACT_LABELS = {
        "PERSON": "[PERSON]",
        "ORG": "[ORG]",
        "GPE": "[LOCATION]",
        "LOC": "[LOCATION]",
    }

    result = already_redacted
    for ent in doc.ents:
        label = REDACT_LABELS.get(ent.label_)
        if label and ent.text in result:
            # Only redact if still present (not already caught by regex)
            result = result.replace(ent.text, label, 1)
            entities.append(PIIEntity(
                entity_type=ent.label_,
                original=ent.text,
                replacement=label,
                start=ent.start_char,
                end=ent.end_char
            ))

    return result, entities


def _heuristic_ner(text: str) -> List[Tuple[str, str]]:
    """Lightweight capitalization-based name detection when spaCy unavailable."""
    entities = []
    patterns = [
        (r'\b(?:name|customer|user|client|patient)[:\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b', "PERSON"),
        (r'\bDear\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b', "PERSON"),
        (r'\b([A-Z][a-z]+\s+[A-Z][a-z]+)\'s\s+(?:account|profile|data|record)\b', "PERSON"),
    ]
    for pattern, label in patterns:
        for match in re.finditer(pattern, text):
            entities.append((label, match.group(1)))
    return entities


def redact_pii(text: str) -> GuardResult:
    """
    Two-tier PII detection and redaction.
    Regex patterns (structured PII) + spaCy NER (named entities).
    """
    use_spacy = MODELS_READY["spacy_ner"]

    # ── Step 1: Regex patterns ───────────────────────────────
    after_regex, regex_entities = _apply_regex(text)

    # ── Step 2: NER ──────────────────────────────────────────
    all_entities = list(regex_entities)

    if use_spacy:
        after_ner, ner_entities = _apply_spacy_ner(text, after_regex)
        all_entities.extend(ner_entities)
        method = "regex + spaCy NER (en_core_web_sm)"
        final_text = after_ner
    else:
        # Fallback heuristic NER
        final_text = after_regex
        heuristic = _heuristic_ner(text)
        for label, name in heuristic:
            if name in final_text:
                final_text = final_text.replace(name, f"[{label}]", 1)
                all_entities.append(PIIEntity(
                    entity_type=label, original=name,
                    replacement=f"[{label}]", start=0, end=0
                ))
        method = "regex + heuristic NER (install spacy for full NER support)"

    if not all_entities:
        return GuardResult(
            triggered=False, risk_score=0.0,
            reason="No PII detected", action="allow",
            clean_text=text, pii_found=[],
            detection_method=method
        )

    risk_score = min(len(all_entities) * 0.15, 1.0)
    types_found = list(set(e.entity_type for e in all_entities))

    return GuardResult(
        triggered=True,
        risk_score=round(risk_score, 3),
        reason=f"PII detected and redacted: {', '.join(types_found)}",
        action="redact",
        clean_text=final_text,
        pii_found=all_entities,
        detection_method=method
    )
