"""
Provider abstraction over the text generator. Two backends, one interface,
so the chain-generation protocol can be re-run unchanged on a second model.

Every backend exposes:

    generate(prompt: str, temperature: float, model: str) -> GenerationResult

GenerationResult carries the text plus enough bookkeeping (token counts,
wall time, model id, temperature) that every output file can record exactly
what produced it, per the mission's "record both in every output file"
requirement.

Rate limiting: Gemini's free-tier quota is per PROJECT, not per API key, so
concurrent requests from a single key can still trip 429s under a thread
pool. Each backend retries on 429 (and generic transient errors) with
exponential backoff plus jitter, capped at MAX_RETRIES.
"""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass
from typing import Optional

import config


@dataclass
class GenerationResult:
    text: str
    model: str
    temperature: float
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    wall_time_seconds: float
    retries: int


class RateLimitError(Exception):
    """Raised when a backend exhausts its retry budget on 429s."""


def _is_retryable(exc: Exception) -> bool:
    msg = str(exc).lower()
    if "429" in msg or "rate limit" in msg or "resource_exhausted" in msg:
        return True
    if "503" in msg or "unavailable" in msg or "timeout" in msg or "timed out" in msg:
        return True
    return False


def _retry_with_backoff(fn, *, max_retries=None, base=None, cap=None):
    """
    Call fn() with exponential backoff + jitter on retryable errors.
    Returns (result, retries_used). Raises the last exception if the
    retry budget is exhausted.
    """
    max_retries = config.MAX_RETRIES if max_retries is None else max_retries
    base = config.BASE_BACKOFF_SECONDS if base is None else base
    cap = config.MAX_BACKOFF_SECONDS if cap is None else cap

    attempt = 0
    while True:
        try:
            return fn(), attempt
        except Exception as exc:  # noqa: BLE001 - deliberately broad, re-raised on give-up
            if not _is_retryable(exc) or attempt >= max_retries:
                raise
            sleep_s = min(cap, base * (2 ** attempt)) * (0.5 + random.random())
            time.sleep(sleep_s)
            attempt += 1


class GeminiProvider:
    name = "gemini"

    def __init__(self, api_key: Optional[str] = None):
        from google import genai

        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError(
                "GEMINI_API_KEY not set. Export it before running the generator."
            )
        self._client = genai.Client(api_key=key)

    def generate(
        self,
        prompt: str,
        temperature: float = config.GENERATION_TEMPERATURE,
        model: str = config.GEMINI_MODEL,
    ) -> GenerationResult:
        from google.genai import types

        def _call():
            t0 = time.monotonic()
            resp = self._client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=temperature),
            )
            elapsed = time.monotonic() - t0
            return resp, elapsed

        (resp, elapsed), retries = _retry_with_backoff(_call)

        text = (resp.text or "").strip()

        prompt_tokens = None
        completion_tokens = None
        usage = getattr(resp, "usage_metadata", None)
        if usage is not None:
            prompt_tokens = getattr(usage, "prompt_token_count", None)
            completion_tokens = getattr(usage, "candidates_token_count", None)

        return GenerationResult(
            text=text,
            model=model,
            temperature=temperature,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            wall_time_seconds=elapsed,
            retries=retries,
        )


class ClaudeProvider:
    name = "claude"

    def __init__(self, api_key: Optional[str] = None):
        import anthropic

        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set. Export it before running the generator."
            )
        self._client = anthropic.Anthropic(api_key=key)

    def generate(
        self,
        prompt: str,
        temperature: float = config.GENERATION_TEMPERATURE,
        model: str = config.CLAUDE_MODEL,
    ) -> GenerationResult:
        def _call():
            t0 = time.monotonic()
            resp = self._client.messages.create(
                model=model,
                max_tokens=200,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            elapsed = time.monotonic() - t0
            return resp, elapsed

        (resp, elapsed), retries = _retry_with_backoff(_call)

        text = "".join(
            block.text for block in resp.content if getattr(block, "type", None) == "text"
        ).strip()

        return GenerationResult(
            text=text,
            model=model,
            temperature=temperature,
            prompt_tokens=getattr(resp.usage, "input_tokens", None),
            completion_tokens=getattr(resp.usage, "output_tokens", None),
            wall_time_seconds=elapsed,
            retries=retries,
        )


_PROVIDERS = {
    "gemini": GeminiProvider,
    "claude": ClaudeProvider,
}


def get_provider(name: str = config.PROVIDER):
    if name not in _PROVIDERS:
        raise ValueError(f"Unknown provider {name!r}. Choices: {list(_PROVIDERS)}")
    return _PROVIDERS[name]()
