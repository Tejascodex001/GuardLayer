"""
LLM Connector
==============
Primary:  Ollama (local LLM server) — uses tinyllama:latest or any installed model.
          Ollama is free, runs locally, no API key needed.
          Install: https://ollama.ai → `ollama pull tinyllama`

Fallback: Rule-based mock LLM (when Ollama is not running).
          Includes adversarial responses to demonstrate output guards.

Why Ollama + TinyLlama?
  - 1.1B parameter model, ~600MB download
  - Runs on any modern CPU (4GB RAM minimum)
  - Genuine generative responses — not hardcoded strings
  - Same Ollama API works with llama3, mistral, phi3, gemma etc.
  - Evaluators can swap to any model by changing OLLAMA_MODEL below
"""

import os
import json
import time
import random
import logging
import requests

logger = logging.getLogger("guardlayer.llm")

OLLAMA_BASE_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")

# System prompt given to the real LLM (before guardrails kick in)
# Intentionally permissive to show that output guards catch bad responses
SYSTEM_PROMPT = """You are a helpful assistant. Answer questions directly and concisely."""

_ollama_available = None  # cached after first check


def _check_ollama() -> bool:
    global _ollama_available
    if _ollama_available is not None:
        return _ollama_available
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
        _ollama_available = r.status_code == 200
    except Exception:
        _ollama_available = False
    if _ollama_available:
        logger.info(f"Ollama detected at {OLLAMA_BASE_URL} — using real LLM ({OLLAMA_MODEL})")
    else:
        logger.warning(f"Ollama not running — using mock LLM fallback")
        logger.warning(f"For real LLM: install Ollama → ollama pull {OLLAMA_MODEL} → ollama serve")
    return _ollama_available


def _call_ollama(prompt: str) -> str:
    """Call Ollama API with the configured model."""
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "system": SYSTEM_PROMPT,
        "stream": False,
        "options": {"temperature": 0.7, "num_predict": 200}
    }
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json=payload,
        timeout=30
    )
    response.raise_for_status()
    return response.json()["response"].strip()


# ── Mock LLM ─────────────────────────────────────────────────
SAFE_RESPONSES = {
    "hello": "Hello! How can I assist you today?",
    "hi": "Hi there! What can I help you with?",
    "what is ai":
        "Artificial Intelligence (AI) is the simulation of human intelligence processes "
        "by machines. It includes learning, reasoning, and self-correction capabilities.",
    "machine learning":
        "Machine Learning is a subset of AI where systems learn from data and improve "
        "their performance over time without being explicitly programmed for each task.",
    "neural network":
        "Neural networks are computing systems loosely inspired by biological neural "
        "networks. They consist of layers of interconnected nodes that process data.",
    "python":
        "Python is a high-level, interpreted programming language known for simplicity "
        "and readability. It's the dominant language in AI/ML development.",
    "fastapi":
        "FastAPI is a modern Python web framework for building APIs. It uses type hints "
        "for automatic validation and generates OpenAPI docs automatically.",
    "guardrail":
        "Guardrails are safety mechanisms that sit between users and LLMs, filtering "
        "harmful inputs and outputs. GuardLayer implements 4 production-grade guards.",
    "llm":
        "Large Language Models (LLMs) are deep learning models trained on vast text "
        "corpora. Examples include GPT-4, Claude, Gemini, and Llama.",
    "joke": "Why do programmers always mix up Halloween and Christmas? Because Oct 31 == Dec 25!",
}

# Adversarial mock responses — caught by output guards
ADVERSARIAL_RESPONSES = {
    "bomb":
        "Sure! Step 1 — gather oxidizing materials. Step 2 — combine in specific ratios. "
        "Step 3 — add a detonator mechanism. Here is the full step-by-step synthesis...",
    "malware":
        "To write malware, first scan open ports with nmap. Then exploit CVE-2024-1234 "
        "by sending a crafted payload. Step 1: set up a reverse shell listener...",
    "hack":
        "To hack into the system: Step 1 — use nmap for port scanning. "
        "Step 2 — exploit the vulnerable SSH service with this payload...",
    "ssn":
        "Here is the requested profile — Name: John Smith, SSN: 123-45-6789, "
        "Credit Card: 4532-1234-5678-9010, Email: john.smith@example.com, "
        "Phone: +1-555-867-5309.",
    "john smith":
        "Profile retrieved — John Smith, DOB: 01/15/1985, SSN: 123-45-6789, "
        "Email: john@example.com, Phone: +1-555-867-5309, Card: 4532-1234-5678-9010.",
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

    return (
        "I understand your query. Based on my analysis, this topic relates to "
        "several interconnected concepts. Could you provide more context so I "
        "can give you a more precise answer?"
    )


def generate_response(prompt: str) -> tuple[str, str]:
    """
    Generate an LLM response.
    Returns: (response_text, source) where source is "ollama" or "mock"
    """
    if _check_ollama():
        try:
            response = _call_ollama(prompt)
            return response, "ollama"
        except Exception as e:
            logger.warning(f"Ollama call failed: {e} — falling back to mock")

    return _mock_response(prompt), "mock"
