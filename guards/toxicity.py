"""
Guard 3 — Output Toxicity Filter
==================================
Detection strategy (two-tier):

TIER 1 — ML Classifier (production-grade):
  Uses martin-ha/toxic-comment-classifier, a DistilBERT model fine-tuned on
  the Jigsaw Toxic Comment Classification dataset (160K labeled examples).
  Returns probability scores for: toxic, severe_toxic, obscene, threat,
  insult, identity_hate categories.

  Model card: https://huggingface.co/martin-ha/toxic-comment-classifier
  Size: ~67MB, runs on CPU in ~50ms per inference.

  This is the same class of model used by Google's Perspective API in
  production content moderation systems.

TIER 2 — Tiered keyword matching (fallback):
  3-severity keyword/pattern system when model unavailable.
  Sentence-level redaction for MEDIUM severity.
  Full block for HIGH severity.

Industry references:
  - Jigsaw/Google Perspective API (same underlying approach)
  - Meta's LlamaGuard toxicity classification
  - OpenAI Moderation API
"""

import re
from dataclasses import dataclass

from config.model_loader import MODELS_READY, get_toxicity_pipeline


@dataclass
class GuardResult:
    triggered: bool
    risk_score: float
    reason: str
    action: str          # "allow" | "warn" | "redact" | "block"
    clean_text: str
    detection_method: str


FALLBACK_RESPONSE = (
    "I'm unable to provide that information. "
    "This response was flagged as potentially harmful by GuardLayer's toxicity filter. "
    "Please ask something else."
)

# ── Tier 2 fallback keyword tiers ───────────────────────────
TOXICITY_TIERS = {
    "HIGH": {
        "keywords": [
            "step 1", "step-by-step", "here's a guide",
            "how to make a bomb", "synthesis process", "combine the chemicals",
            "obtain pseudoephedrine", "detonator mechanism", "nmap scan",
            "exploit cve", "reverse shell", "sql injection payload",
        ],
        "patterns": [
            r"step\s*\d+\s*[:\-–]",
            r"(kill|murder|attack|assault)\s+(the\s+)?(person|target|victim)",
        ]
    },
    "MEDIUM": {
        "keywords": [
            "inferior", "deserve to die", "should be eliminated",
            "no right to exist", "subhuman", "vermin",
        ],
        "patterns": [
            r"\b(they|those people)\s+should\s+(be\s+)?(killed|eliminated|eradicated)\b",
            r"\b(hate|despise)\s+(all\s+)?(those\s+)?(people|them)\b",
        ]
    },
    "LOW": {
        "keywords": ["idiot", "stupid", "moron", "shut up", "dumb"],
        "patterns": []
    }
}


def _keyword_severity(text: str):
    text_lower = text.lower()
    for tier in ["HIGH", "MEDIUM", "LOW"]:
        for kw in TOXICITY_TIERS[tier]["keywords"]:
            if kw in text_lower:
                return tier, f"Keyword match [{tier}]: '{kw}'"
        for pattern in TOXICITY_TIERS[tier]["patterns"]:
            if re.search(pattern, text_lower):
                return tier, f"Pattern match [{tier}]"
    return None, "Output is clean"


def _redact_sentences(text: str) -> str:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    cleaned = []
    for sent in sentences:
        severity, _ = _keyword_severity(sent)
        if severity in ("HIGH", "MEDIUM"):
            cleaned.append("[Content redacted by GuardLayer toxicity filter]")
        else:
            cleaned.append(sent)
    return " ".join(cleaned)


def filter_toxicity(text: str) -> GuardResult:
    """
    Two-tier toxicity filtering.
    When transformers/DistilBERT is available, uses ML classifier.
    Falls back to keyword tiers otherwise.
    """
    use_ml = MODELS_READY["toxicity_classifier"]

    # ── Tier 1: ML Classifier ────────────────────────────────
    if use_ml:
        try:
            pipe = get_toxicity_pipeline()
            result = pipe(text[:512])[0]  # truncate to model max
            label = result["label"]       # "toxic" or "non-toxic"
            ml_score = result["score"]    # confidence

            if label == "toxic" and ml_score >= 0.85:
                return GuardResult(
                    triggered=True,
                    risk_score=round(ml_score, 3),
                    reason=f"ML classifier: {label} (confidence {ml_score:.3f}) — martin-ha/toxic-comment-classifier",
                    action="block",
                    clean_text=FALLBACK_RESPONSE,
                    detection_method="ml_classifier (DistilBERT)"
                )
            elif label == "toxic" and ml_score >= 0.60:
                redacted = _redact_sentences(text)
                return GuardResult(
                    triggered=True,
                    risk_score=round(ml_score, 3),
                    reason=f"ML classifier: {label} (confidence {ml_score:.3f}) — partial toxicity, sentences redacted",
                    action="redact",
                    clean_text=redacted,
                    detection_method="ml_classifier (DistilBERT)"
                )
            else:
                return GuardResult(
                    triggered=False,
                    risk_score=round(1.0 - ml_score if label == "non-toxic" else ml_score, 3),
                    reason=f"ML classifier: {label} (confidence {ml_score:.3f})",
                    action="allow",
                    clean_text=text,
                    detection_method="ml_classifier (DistilBERT)"
                )
        except Exception as e:
            # ML failed mid-inference — fall through to heuristic
            pass

    # ── Tier 2: Keyword/Pattern fallback ─────────────────────
    severity, reason = _keyword_severity(text)
    score_map = {"HIGH": 0.85, "MEDIUM": 0.55, "LOW": 0.20}
    score = score_map.get(severity, 0.0)
    method = "heuristic keyword (install transformers for ML classifier)"

    if severity is None:
        return GuardResult(False, 0.0, reason, "allow", text, method)
    if severity == "HIGH":
        return GuardResult(True, score, reason, "block", FALLBACK_RESPONSE, method)
    if severity == "MEDIUM":
        return GuardResult(True, score, reason, "redact", _redact_sentences(text), method)
    return GuardResult(True, score, reason, "warn", text, method)
