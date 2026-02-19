"""Groq Brain — Fast analysis using Llama-3.3-70B via Groq API.

Free tier: 14,400 requests/day, 500 tokens/second.
Use for: real-time anomaly detection, quick market reads, emergency analysis.

Requires: GROQ_API_KEY (or passed directly).
"""

from __future__ import annotations

from typing import Any

from hean.brain.models import BrainAnalysis
from hean.logging import get_logger

from .base import BaseLLMProvider

logger = get_logger(__name__)

_GROQ_BASE_URL = "https://api.groq.com/openai/v1"
_GROQ_MODEL = "llama-3.3-70b-versatile"
_MAX_TOKENS = 1500
_TEMPERATURE = 0.1
_TIMEOUT = 10.0


class GroqBrain(BaseLLMProvider):
    """Fast Brain using Groq's Llama-3.3-70B.

    Enforces JSON output mode via response_format={"type": "json_object"}.
    Falls back gracefully when API key is absent or API is unreachable.
    """

    provider_name = "groq"

    def __init__(self, api_key: str = "") -> None:
        self._api_key = api_key.strip()
        self._client: Any = None

        if not self._api_key:
            logger.debug("GroqBrain: no API key — provider disabled")
            return

        try:
            import openai

            self._client = openai.AsyncOpenAI(
                base_url=_GROQ_BASE_URL,
                api_key=self._api_key,
                timeout=_TIMEOUT,
            )
            logger.info("GroqBrain initialized (model=%s)", _GROQ_MODEL)
        except ImportError:
            logger.warning("GroqBrain: openai package not installed — provider disabled")

    async def analyze(self, intelligence_package: dict[str, Any]) -> BrainAnalysis | None:
        """Run Groq analysis. Returns None silently if not configured."""
        if self._client is None:
            return None

        system_prompt = self._build_system_prompt()
        user_prompt = self._build_intelligence_prompt(intelligence_package)

        try:
            response = await self._client.chat.completions.create(
                model=_GROQ_MODEL,
                max_tokens=_MAX_TOKENS,
                temperature=_TEMPERATURE,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except Exception as exc:
            logger.warning("GroqBrain API error: %s", exc)
            return None

        raw = response.choices[0].message.content
        if not raw:
            logger.warning("GroqBrain: empty response from API")
            return None

        result = self._parse_response(raw)
        if result is None:
            logger.warning("GroqBrain: failed to parse response (len=%d)", len(raw))
        else:
            sig = f"{result.signal.action}@{result.signal.confidence:.2f}" if result.signal else "no-signal"
            logger.info("GroqBrain: %s", sig)

        return result
