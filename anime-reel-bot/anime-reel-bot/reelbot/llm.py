"""Free LLM access: Groq first (generous free daily quota), Gemini as fallback."""
from __future__ import annotations

import json
import re
import time

import requests

from .config import Config


class LLMError(RuntimeError):
    pass


def _parse_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            raise
        return json.loads(m.group(0))


def _post(url, headers, payload, tries=3):
    last = None
    for k in range(tries):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=90)
            if r.status_code in (429, 500, 502, 503, 504):
                last = f"HTTP {r.status_code}"
                time.sleep(4 * (k + 1))
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:   # never include headers (keys) in the message
            last = type(e).__name__
            time.sleep(3 * (k + 1))
    raise LLMError(f"request failed: {last}")


def _groq(cfg: Config, system: str, user: str, temperature: float) -> dict:
    data = _post(
        "https://api.groq.com/openai/v1/chat/completions",
        {"Authorization": f"Bearer {cfg.groq_key}"},
        {"model": cfg.groq_model, "temperature": temperature, "max_completion_tokens": 3000,
         "reasoning_effort": "low", "response_format": {"type": "json_object"},
         "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]})
    return _parse_json(data["choices"][0]["message"]["content"])


def _gemini(cfg: Config, system: str, user: str, temperature: float) -> dict:
    data = _post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{cfg.gemini_model}:generateContent",
        {"x-goog-api-key": cfg.gemini_key},
        {"systemInstruction": {"parts": [{"text": system}]},
         "contents": [{"role": "user", "parts": [{"text": user}]}],
         "generationConfig": {"temperature": temperature, "responseMimeType": "application/json"}})
    return _parse_json(data["candidates"][0]["content"]["parts"][0]["text"])


def chat_json(cfg: Config, system: str, user: str, temperature: float = 0.8) -> dict:
    errors = []
    for name, key, fn in (("groq", cfg.groq_key, _groq), ("gemini", cfg.gemini_key, _gemini)):
        if not key:
            continue
        try:
            return fn(cfg, system, user, temperature)
        except Exception as e:  # noqa: BLE001 - try the next provider
            errors.append(f"{name}: {type(e).__name__}: {str(e)[:120]}")
    raise LLMError("no LLM available: " + "; ".join(errors or ["no API key set"]))
