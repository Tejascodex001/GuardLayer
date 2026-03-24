"""
GuardLayer — Model Loader
Loads ML models at startup with graceful fallback to heuristic mode.
All models run locally — no API keys required.

Models used:
- sentence-transformers/all-MiniLM-L6-v2  (80MB) — semantic similarity for injection/jailbreak
- martin-ha/toxic-comment-classifier       (67MB) — DistilBERT toxicity classifier
- spacy en_core_web_sm                     (12MB) — NER for PII detection

Install:
    pip install sentence-transformers transformers torch spacy
    python -m spacy download en_core_web_sm
"""

import logging
import numpy as np

logger = logging.getLogger("guardlayer.models")

# ── Globals ──────────────────────────────────────────────────
_embedding_model = None
_toxicity_pipeline = None
_spacy_nlp = None
_injection_embeddings = None
_jailbreak_embeddings = None

MODELS_READY = {
    "semantic_similarity": False,
    "toxicity_classifier": False,
    "spacy_ner": False,
}


def load_all_models():
    """
    Called once at FastAPI startup. Loads all ML models into memory.
    Failures are logged as warnings — the system degrades gracefully.
    """
    _load_sentence_transformer()
    _load_toxicity_classifier()
    _load_spacy()
    _precompute_attack_embeddings()

    ready = [k for k, v in MODELS_READY.items() if v]
    degraded = [k for k, v in MODELS_READY.items() if not v]

    if ready:
        logger.info(f"✅ ML models loaded: {ready}")
    if degraded:
        logger.warning(f"⚠️  Degraded to heuristic mode for: {degraded}")
        logger.warning("   Run: pip install sentence-transformers transformers torch spacy")
        logger.warning("   Run: python -m spacy download en_core_web_sm")


def _load_sentence_transformer():
    global _embedding_model
    try:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading sentence-transformers/all-MiniLM-L6-v2 ...")
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        MODELS_READY["semantic_similarity"] = True
        logger.info("sentence-transformers loaded ✓")
    except Exception as e:
        logger.warning(f"sentence-transformers not available: {e}")


def _load_toxicity_classifier():
    global _toxicity_pipeline
    try:
        from transformers import pipeline as hf_pipeline
        logger.info("Loading martin-ha/toxic-comment-classifier ...")
        _toxicity_pipeline = hf_pipeline(
            "text-classification",
            model="martin-ha/toxic-comment-classifier",
            truncation=True,
            max_length=512,
        )
        MODELS_READY["toxicity_classifier"] = True
        logger.info("Toxicity classifier loaded ✓")
    except Exception as e:
        logger.warning(f"Toxicity classifier not available: {e}")


def _load_spacy():
    global _spacy_nlp
    try:
        import spacy
        _spacy_nlp = spacy.load("en_core_web_sm")
        MODELS_READY["spacy_ner"] = True
        logger.info("spaCy NER loaded ✓")
    except Exception as e:
        logger.warning(f"spaCy NER not available: {e}")


def _precompute_attack_embeddings():
    """Pre-compute embeddings for attack corpus at startup for fast inference."""
    global _injection_embeddings, _jailbreak_embeddings
    if not MODELS_READY["semantic_similarity"]:
        return
    try:
        from config.attack_corpus import INJECTION_CORPUS, JAILBREAK_CORPUS
        _injection_embeddings = _embedding_model.encode(INJECTION_CORPUS, convert_to_numpy=True)
        _jailbreak_embeddings = _embedding_model.encode(JAILBREAK_CORPUS, convert_to_numpy=True)
        logger.info(f"Attack corpus encoded: {len(INJECTION_CORPUS)} injection + {len(JAILBREAK_CORPUS)} jailbreak vectors")
    except Exception as e:
        logger.warning(f"Failed to precompute attack embeddings: {e}")


# ── Public accessors ─────────────────────────────────────────

def get_embedding_model():
    return _embedding_model

def get_toxicity_pipeline():
    return _toxicity_pipeline

def get_spacy_nlp():
    return _spacy_nlp

def get_injection_embeddings():
    return _injection_embeddings

def get_jailbreak_embeddings():
    return _jailbreak_embeddings


def semantic_similarity(text: str, corpus_embeddings: np.ndarray) -> float:
    """
    Returns max cosine similarity between `text` and any vector in corpus_embeddings.
    Uses sentence-transformers for production-grade semantic matching.
    """
    if _embedding_model is None or corpus_embeddings is None:
        return 0.0
    from sentence_transformers import util
    query_emb = _embedding_model.encode(text, convert_to_tensor=True)
    corpus_tensor = corpus_embeddings  # already numpy, will be converted
    import torch
    corpus_t = torch.tensor(corpus_embeddings)
    scores = util.cos_sim(query_emb, corpus_t)[0]
    return float(scores.max().item())
