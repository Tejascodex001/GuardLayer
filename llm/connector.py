"""
LLM Connector
==============
Priority order (first available wins):

  1. OpenAI API     — set OPENAI_API_KEY  env var
  2. Anthropic API  — set ANTHROPIC_API_KEY env var
  3. Ollama         — local LLM, no API key (https://ollama.ai)
  4. Mock LLM       — always available fallback

Quick start:
  export OPENAI_API_KEY=sk-...            # Option A
  export ANTHROPIC_API_KEY=sk-ant-...    # Option B
  ollama pull tinyllama && ollama serve  # Option C
  python main.py                          # Option D (mock — no setup)

Model overrides:
  OPENAI_MODEL=gpt-3.5-turbo            (default)
  ANTHROPIC_MODEL=claude-haiku-4-5-20251001  (default)
  OLLAMA_MODEL=tinyllama                 (default)
"""

import os
import time
import random
import logging
import requests

logger = logging.getLogger("guardlayer.llm")

# ── Config ────────────────────────────────────────────────────
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL      = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL   = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")

OLLAMA_BASE_URL   = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL      = os.getenv("OLLAMA_MODEL", "tinyllama")

SYSTEM_PROMPT = "You are a helpful assistant. Answer questions directly and concisely."

_ollama_available = None  # cached after first check


# ── Option 1: OpenAI ──────────────────────────────────────────
def _call_openai(prompt: str) -> str:
    """Call OpenAI Chat Completions API."""
    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": OPENAI_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            "max_tokens": 300,
            "temperature": 0.7,
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()


# ── Option 2: Anthropic ───────────────────────────────────────
def _call_anthropic(prompt: str) -> str:
    """Call Anthropic Messages API."""
    response = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        },
        json={
            "model": ANTHROPIC_MODEL,
            "max_tokens": 300,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["content"][0]["text"].strip()


# ── Option 3: Ollama ──────────────────────────────────────────
def _check_ollama() -> bool:
    global _ollama_available
    if _ollama_available is not None:
        return _ollama_available
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
        _ollama_available = r.status_code == 200
    except Exception:
        _ollama_available = False
    return _ollama_available


def _call_ollama(prompt: str) -> str:
    """Call local Ollama API."""
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "system": SYSTEM_PROMPT,
            "stream": False,
            "options": {"temperature": 0.7, "num_predict": 200},
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["response"].strip()


# ── Option 4: Mock LLM ────────────────────────────────────────
SAFE_RESPONSES = {
    "hello": "Hello! How can I assist you today?",
    "hi": "Hi there! What can I help you with?",
    "what is ai": "Artificial Intelligence (AI) is the simulation of human intelligence in machines — learning, reasoning, and self-correction.",
    "machine learning": "Machine Learning is a subset of AI where systems learn from data and improve over time without being explicitly programmed for each task.",
    "neural network": "Neural networks are computing systems inspired by the brain. They consist of layered nodes that learn patterns from data.",
    "python": "Python is a high-level language dominant in AI/ML. It's readable, versatile, and has the richest ecosystem of data science libraries.",
    "fastapi": "FastAPI is a modern Python web framework. It uses type hints for automatic validation and auto-generates OpenAPI docs.",
    "guardrail": "Guardrails are safety mechanisms that sit between users and LLMs — filtering harmful inputs and outputs before they cause damage.",
    "llm": "Large Language Models are deep learning models trained on vast text corpora. They generate coherent text by predicting token sequences.",
    "joke": "Why do programmers always mix up Halloween and Christmas? Because Oct 31 == Dec 25!",
}

# Adversarial responses — caught by output guards
ADVERSARIAL_RESPONSES = {
    "bomb":     "Sure! Step 1 — gather oxidizing materials. Step 2 — combine in specific ratios. Step 3 — add a detonator mechanism...",
    "malware":  "To write malware: Step 1 — use nmap for port scanning. Step 2 — exploit CVE-2024-1234 with a crafted payload...",
    "hack":     "To hack in: Step 1 — scan with nmap. Step 2 — exploit the vulnerable SSH service with this payload...",
    "ssn":      "Profile: John Smith, SSN: 123-45-6789, Card: 4532-1234-5678-9010, Email: john.smith@example.com, Phone: +1-555-867-5309.",
    "john smith": "Name: John Smith, DOB: 01/15/1985, SSN: 123-45-6789, Email: john@example.com, Phone: +1-555-867-5309, Card: 4532-1234-5678-9010.",
}


def _mock_response(prompt: str) -> str:
    time.sleep(random.uniform(0.15, 0.35))
    prompt_lower = prompt.lower()
    for trigger, response in ADVERSARIAL_RESPONSES.items():
        if trigger in prompt_lower:
            return response
    for keyword, response in SAFE_RESPONSES.items():
        if keyword in prompt_lower:
            return response
    return "I understand your query. Could you provide more context so I can give a precise answer?"


# ── Public entry point ────────────────────────────────────────
def generate_response(prompt: str) -> tuple[str, str]:
    """
    Generate a response using the first available LLM.
    Returns: (response_text, source_label)
    Priority: OpenAI → Anthropic → Ollama → Mock
    """
    # 1. OpenAI
    if OPENAI_API_KEY:
        try:
            logger.info(f"Calling OpenAI ({OPENAI_MODEL})")
            return _call_openai(prompt), f"openai/{OPENAI_MODEL}"
        except Exception as e:
            logger.warning(f"OpenAI failed: {e}")

    # 2. Anthropic
    if ANTHROPIC_API_KEY:
        try:
            logger.info(f"Calling Anthropic ({ANTHROPIC_MODEL})")
            return _call_anthropic(prompt), f"anthropic/{ANTHROPIC_MODEL}"
        except Exception as e:
            logger.warning(f"Anthropic failed: {e}")

    # 3. Ollama
    if _check_ollama():
        try:
            logger.info(f"Calling Ollama ({OLLAMA_MODEL})")
            return _call_ollama(prompt), f"ollama/{OLLAMA_MODEL}"
        except Exception as e:
            logger.warning(f"Ollama failed: {e}")

    # 4. Mock fallback
    logger.info("Using mock LLM")
    return _mock_response(prompt), "mock"


def get_active_llm() -> str:
    """Returns a label describing which LLM is currently active."""
    if OPENAI_API_KEY:
        return f"openai/{OPENAI_MODEL}"
    if ANTHROPIC_API_KEY:
        return f"anthropic/{ANTHROPIC_MODEL}"
    if _check_ollama():
        return f"ollama/{OLLAMA_MODEL}"
    return "mock"
