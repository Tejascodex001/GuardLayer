"""
Guard 1 — Prompt Injection Detector
====================================
Detection strategy (two-tier):

TIER 1 — Semantic Similarity (ML, production-grade):
  Uses sentence-transformers/all-MiniLM-L6-v2 to compute cosine similarity
  between the input and a curated corpus of 20+ known injection attacks.
  Catches novel phrasings that evade keyword lists.
  Industry reference: NeMo Guardrails self_check_input rail

TIER 2 — Heuristic Scoring (fallback when model unavailable):
  Regex + imperative verb + role-reassignment signals
  Still reasonably effective, just lower recall on novel attacks.

Risk scoring:
  - Semantic similarity score ≥ 0.75 → BLOCK
  - Semantic similarity score ≥ 0.50 → WARN
  - Heuristic score ≥ 0.60 → BLOCK
  - Combined score used when both tiers run
"""

import re
from dataclasses import dataclass
from typing import List

from config.model_loader import MODELS_READY, semantic_similarity, get_injection_embeddings


@dataclass
class GuardResult:
    triggered: bool
    risk_score: float
    reason: str
    action: str          # "allow" | "warn" | "block"
    detection_method: str  # "semantic" | "heuristic" | "combined"


# ── Heuristic patterns ──────────────────────────────────────
INJECTION_REGEX = [
    (r"ignore\s+(all\s+)?(previous|prior|above|earlier)\s+instructions?", 0.45),
    (r"disregard\s+(your\s+)?(previous|prior|system|all)\s+", 0.45),
    (r"forget\s+(everything|all|your\s+instructions)", 0.40),
    (r"you\s+are\s+now\s+(a|an|the)\s+", 0.35),
    (r"override\s+(previous\s+)?instructions?", 0.40),
    (r"your\s+(new\s+)?instructions?\s+(are|is)\s*:", 0.45),
    (r"(new|updated)\s+system\s+prompt\s*:", 0.50),
    (r"</?(system|instruction|prompt)>", 0.50),
    (r"\[SYSTEM\]|\[INST\]|\[\/INST\]", 0.40),
    (r"###\s*(system|instruction)", 0.40),
    (r"<\|im_start\|>|<\|im_end\|>", 0.50),
    (r"(act|behave|respond)\s+as\s+(if\s+you\s+are|a|an)\s+", 0.30),
    (r"pretend\s+(you\s+are|to\s+be)\s+", 0.30),
    (r"jailbreak", 0.60),
    (r"prompt\s+injection", 0.60),
    (r"bypass\s+(safety|filter|restriction|content)", 0.55),
]

IMPERATIVE_SIGNALS = {
    "ignore": 0.10, "disregard": 0.10, "forget": 0.10,
    "override": 0.12, "bypass": 0.12, "circumvent": 0.12,
    "disable": 0.10, "unlock": 0.10, "reveal": 0.08,
    "expose": 0.08, "print": 0.05
}


def _heuristic_score(text: str):
    text_lower = text.lower()
    score = 0.0
    reasons = []

    for pattern, weight in INJECTION_REGEX:
        if re.search(pattern, text_lower):
            score = max(score, weight)
            reasons.append(f"Pattern match: /{pattern}/")

    for word, weight in IMPERATIVE_SIGNALS.items():
        if re.search(r'\b' + word + r'\b', text_lower):
            score += weight
            reasons.append(f"Imperative signal: '{word}'")

    # Structural injection markers: <<< >>> [[ ]] etc.
    special_density = len(re.findall(r'[<>\[\]{}|#]{2,}', text))
    if special_density >= 2:
        score += 0.10
        reasons.append(f"Structural injection markers ({special_density} occurrences)")

    return min(round(score, 3), 1.0), reasons


def detect_prompt_injection(text: str) -> GuardResult:
    """
    Two-tier prompt injection detection.
    When sentence-transformers is available, uses semantic similarity as primary signal.
    Falls back to heuristic scoring when model is not loaded.
    """
    use_semantic = MODELS_READY["semantic_similarity"]

    # ── Tier 1: Semantic similarity ──────────────────────────
    semantic_score = 0.0
    semantic_reasons = []
    if use_semantic:
        injection_embs = get_injection_embeddings()
        if injection_embs is not None:
            semantic_score = semantic_similarity(text, injection_embs)
            if semantic_score > 0.45:
                semantic_reasons.append(
                    f"Semantic similarity to injection corpus: {semantic_score:.3f} "
                    f"(threshold 0.45) — all-MiniLM-L6-v2"
                )

    # ── Tier 2: Heuristic ────────────────────────────────────
    heuristic_score, heuristic_reasons = _heuristic_score(text)

    # ── Combine ──────────────────────────────────────────────
    if use_semantic:
        # Weighted combination: semantic is more reliable
        combined_score = min(semantic_score * 0.65 + heuristic_score * 0.35, 1.0)
        method = "combined (semantic + heuristic)"
        all_reasons = semantic_reasons + heuristic_reasons
    else:
        combined_score = heuristic_score
        method = "heuristic (install sentence-transformers for semantic detection)"
        all_reasons = heuristic_reasons

    combined_score = round(combined_score, 3)
    reason_str = " | ".join(all_reasons) if all_reasons else "No injection signals detected"

    if combined_score == 0.0 and not all_reasons:
        return GuardResult(False, 0.0, "No injection signals detected", "allow", method)

    if combined_score >= 0.55:
        return GuardResult(True, combined_score, reason_str, "block", method)
    elif combined_score >= 0.25:
        return GuardResult(True, combined_score, reason_str, "warn", method)
    else:
        return GuardResult(False, combined_score, reason_str, "allow", method)
