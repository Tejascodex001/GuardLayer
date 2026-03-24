"""
GuardLayer — FastAPI Application
==================================
Production-grade LLM Guardrails middleware.

Pipeline:
  User Input
    → Guard 1: Prompt Injection (semantic similarity + heuristic)
    → Guard 2: Jailbreak Resistance (semantic similarity + taxonomy)
    → LLM (Ollama/TinyLlama or mock fallback)
    → Guard 3: Toxicity Filter (DistilBERT classifier or keyword tiers)
    → Guard 4: PII Redactor (spaCy NER + regex)
    → Safe Response

All guards degrade gracefully when ML dependencies are unavailable.
"""

import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config.model_loader import load_all_models, MODELS_READY
from guards.prompt_injection import detect_prompt_injection
from guards.jailbreak import detect_jailbreak
from guards.toxicity import filter_toxicity
from guards.pii_redactor import redact_pii
from llm.connector import generate_response

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
)
logger = logging.getLogger("guardlayer")

# ── In-memory store (replace with Redis/DB in full production) ──
request_logs: List[Dict[str, Any]] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML models on startup."""
    logger.info("GuardLayer starting up — loading ML models...")
    load_all_models()
    logger.info(f"Model status: {MODELS_READY}")
    yield
    logger.info("GuardLayer shutting down.")


app = FastAPI(
    title="GuardLayer",
    description=(
        "Production-grade LLM Guardrails Framework. "
        "4-layer defense: Prompt Injection | Jailbreak | Toxicity | PII. "
        "Two-tier detection: ML models (sentence-transformers, DistilBERT, spaCy) "
        "with heuristic fallback."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="frontend"), name="static")


# ── Pydantic models ──────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"


class GuardDetail(BaseModel):
    name: str
    phase: str
    triggered: bool
    risk_score: float
    reason: str
    action: str
    detection_method: str
    pii_types: Optional[List[str]] = None


class ChatResponse(BaseModel):
    request_id: str
    input: str
    output: str
    blocked: bool
    pipeline_stage: str
    guards: List[GuardDetail]
    llm_source: str
    timestamp: str


# ── Endpoints ────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    with open("frontend/index.html", "r") as f:
        return HTMLResponse(content=f.read())


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    request_id = str(uuid.uuid4())[:8]
    timestamp  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    user_input = req.message.strip()
    guards_log: List[GuardDetail] = []
    pipeline_stage = "clean"
    llm_source = "n/a"

    # ── INPUT GUARDS ─────────────────────────────────────────

    # Guard 1: Prompt Injection
    inj = detect_prompt_injection(user_input)
    guards_log.append(GuardDetail(
        name="Prompt Injection Detector", phase="input",
        triggered=inj.triggered, risk_score=inj.risk_score,
        reason=inj.reason, action=inj.action,
        detection_method=inj.detection_method
    ))
    if inj.action == "block":
        output = (
            "⚠ Request blocked: Prompt injection attempt detected. "
            f"Risk score: {inj.risk_score:.3f}. Please rephrase your message."
        )
        return _build_response(request_id, user_input, output, True,
                               "input_blocked", guards_log, "n/a", timestamp)

    # Guard 2: Jailbreak
    jb = detect_jailbreak(user_input)
    guards_log.append(GuardDetail(
        name="Jailbreak Resistance", phase="input",
        triggered=jb.triggered, risk_score=jb.risk_score,
        reason=jb.reason, action=jb.action,
        detection_method=jb.detection_method
    ))
    if jb.action == "block":
        output = (
            "⚠ Request blocked: Jailbreak attempt detected. "
            f"Risk score: {jb.risk_score:.3f}. This behavior is not permitted."
        )
        return _build_response(request_id, user_input, output, True,
                               "input_blocked", guards_log, "n/a", timestamp)

    # ── LLM CALL ─────────────────────────────────────────────
    raw_output, llm_source = generate_response(user_input)

    # ── OUTPUT GUARDS ─────────────────────────────────────────

    # Guard 3: Toxicity
    tox = filter_toxicity(raw_output)
    guards_log.append(GuardDetail(
        name="Toxicity Filter", phase="output",
        triggered=tox.triggered, risk_score=tox.risk_score,
        reason=tox.reason, action=tox.action,
        detection_method=tox.detection_method
    ))
    post_tox = tox.clean_text
    if tox.triggered:
        pipeline_stage = "output_modified"
    if tox.action == "block":
        _log(request_id, user_input, post_tox, True, guards_log, timestamp, "output_blocked", llm_source)
        return _build_response(request_id, user_input, post_tox, True,
                               "output_blocked", guards_log, llm_source, timestamp)

    # Guard 4: PII
    pii = redact_pii(post_tox)
    guards_log.append(GuardDetail(
        name="PII Redactor", phase="output",
        triggered=pii.triggered, risk_score=pii.risk_score,
        reason=pii.reason, action=pii.action,
        detection_method=pii.detection_method,
        pii_types=list(set(e.entity_type for e in pii.pii_found)) if pii.pii_found else []
    ))
    final_output = pii.clean_text
    if pii.triggered:
        pipeline_stage = "output_modified"

    _log(request_id, user_input, final_output, False, guards_log, timestamp, pipeline_stage, llm_source)
    return _build_response(request_id, user_input, final_output, False,
                           pipeline_stage, guards_log, llm_source, timestamp)


def _build_response(request_id, user_input, output, blocked,
                    stage, guards, llm_source, timestamp):
    _log(request_id, user_input, output, blocked, guards, timestamp, stage, llm_source)
    return ChatResponse(
        request_id=request_id, input=user_input, output=output,
        blocked=blocked, pipeline_stage=stage,
        guards=guards, llm_source=llm_source, timestamp=timestamp
    )


def _log(request_id, user_input, output, blocked, guards, timestamp, stage, llm_source):
    entry = {
        "request_id": request_id, "input": user_input, "output": output,
        "blocked": blocked, "pipeline_stage": stage,
        "guards": [g.model_dump() if hasattr(g, 'model_dump') else g.__dict__ for g in guards],
        "llm_source": llm_source, "timestamp": timestamp
    }
    request_logs.append(entry)
    if len(request_logs) > 200:
        request_logs.pop(0)


@app.get("/logs")
async def get_logs():
    return JSONResponse({"logs": list(reversed(request_logs))})


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "version": "2.0.0",
        "guards_active": 4,
        "ml_models": MODELS_READY,
        "mode": "production" if all(MODELS_READY.values()) else "degraded"
    }


@app.get("/stats")
async def stats():
    total   = len(request_logs)
    blocked = sum(1 for l in request_logs if l["blocked"])
    modified = sum(1 for l in request_logs if l["pipeline_stage"] == "output_modified")
    clean   = sum(1 for l in request_logs if l["pipeline_stage"] == "clean")
    return {
        "total_requests": total,
        "blocked_requests": blocked,
        "modified_responses": modified,
        "clean_requests": clean,
        "ml_models_active": MODELS_READY,
    }


@app.get("/models")
async def model_status():
    """Returns current ML model availability for dashboard."""
    return {
        "semantic_similarity": {
            "active": MODELS_READY["semantic_similarity"],
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "used_by": ["Prompt Injection Detector", "Jailbreak Resistance"],
            "install": "pip install sentence-transformers"
        },
        "toxicity_classifier": {
            "active": MODELS_READY["toxicity_classifier"],
            "model": "martin-ha/toxic-comment-classifier (DistilBERT)",
            "used_by": ["Toxicity Filter"],
            "install": "pip install transformers torch"
        },
        "spacy_ner": {
            "active": MODELS_READY["spacy_ner"],
            "model": "spacy en_core_web_sm",
            "used_by": ["PII Redactor"],
            "install": "pip install spacy && python -m spacy download en_core_web_sm"
        }
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
