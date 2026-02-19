"""Ollama Brain — Local LLM for offline / zero-cost operation.

Uses aiohttp to call a local Ollama server. No API costs.
Recommended models: deepseek-r1:14b, qwen3:8b, llama3.3:latest.

Falls back silently (returns None) when Ollama is unreachable.
"""

from __future__ import annotations

from typing import Any

from hean.brain.models import BrainAnalysis
from hean.logging import get_logger

from .base import BaseLLMProvider

logger = get_logger(__name__)

_DEFAULT_URL = "http://localhost:11434"
_DEFAULT_MODEL = "deepseek-r1:14b"
_TIMEOUT_SECONDS = 60.0
_GENERATE_ENDPOINT = "/api/generate"


class OllamaBrain(BaseLLMProvider):
    """Local LLM Brain via Ollama HTTP API.

    Uses the /api/generate endpoint (single-turn, non-streaming).
    No dependency on the openai package — pure aiohttp calls.
    """

    provider_name = "ollama"

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        ollama_url: str = _DEFAULT_URL,
    ) -> None:
        self._model = model.strip() or _DEFAULT_MODEL
        self._base_url = ollama_url.rstrip("/")
        self._generate_url = f"{self._base_url}{_GENERATE_ENDPOINT}"
        logger.info("OllamaBrain configured (url=%s, model=%s)", self._base_url, self._model)

    async def analyze(self, intelligence_package: dict[str, Any]) -> BrainAnalysis | None:
        """Run local Ollama analysis. Returns None silently if Ollama is unreachable."""
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_intelligence_prompt(intelligence_package)
        full_prompt = f"{system_prompt}\n\n{user_prompt}"

        payload: dict[str, Any] = {
            "model": self._model,
            "prompt": full_prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0.1,
                "num_predict": 1500,
            },
        }

        try:
            import aiohttp
        except ImportError:
            logger.warning("OllamaBrain: aiohttp not installed — provider disabled. pip install aiohttp")
            return None

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self._generate_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=_TIMEOUT_SECONDS),
                ) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        logger.warning("OllamaBrain: HTTP %d (body=%r)", resp.status, body[:200])
                        return None
                    data = await resp.json()

        except aiohttp.ClientConnectorError:
            logger.debug("OllamaBrain: cannot connect to Ollama at %s", self._base_url)
            return None
        except Exception as exc:
            logger.warning("OllamaBrain: request error: %s", exc)
            return None

        raw = data.get("response", "")
        if not raw:
            logger.warning("OllamaBrain: empty 'response' field from Ollama")
            return None

        result = self._parse_response(raw)
        if result is None:
            logger.warning("OllamaBrain: failed to parse response (len=%d)", len(raw))
        else:
            sig = f"{result.signal.action}@{result.signal.confidence:.2f}" if result.signal else "no-signal"
            logger.info("OllamaBrain: %s", sig)

        return result
