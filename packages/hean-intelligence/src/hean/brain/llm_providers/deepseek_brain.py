"""DeepSeek Brain — Deep reasoning using DeepSeek-R1.

Primary:  DeepSeek direct API (api.deepseek.com) — $0.55/M tokens, cheapest thinking model.
Fallback: OpenRouter (deepseek/deepseek-r1) — used when deepseek_api_key absent
          but openrouter_api_key is present.

Use for: complex regime analysis, causal reasoning, contradictory signals.

Note: DeepSeek-R1 does NOT support response_format=json_object.
      The model emits <think>...</think> blocks which are stripped before JSON extraction.
"""

from __future__ import annotations

from typing import Any

from hean.brain.models import BrainAnalysis
from hean.logging import get_logger

from .base import BaseLLMProvider

logger = get_logger(__name__)

_DS_BASE_URL = "https://api.deepseek.com"
_DS_MODEL = "deepseek-reasoner"
_OR_BASE_URL = "https://openrouter.ai/api/v1"
_OR_MODEL = "deepseek/deepseek-r1"
_OR_HEADERS = {
    "HTTP-Referer": "https://hean.trading",
    "X-Title": "HEAN Sovereign Brain",
}
_MAX_TOKENS = 2000
_TEMPERATURE = 0.0
_TIMEOUT = 30.0


class DeepSeekBrain(BaseLLMProvider):
    """Deep reasoning Brain using DeepSeek-R1.

    Tries the DeepSeek direct API first; falls back to OpenRouter if the
    direct key is missing. Returns None silently when neither is configured.
    """

    provider_name = "deepseek"

    def __init__(
        self,
        deepseek_api_key: str = "",
        openrouter_api_key: str = "",
    ) -> None:
        self._ds_key = deepseek_api_key.strip()
        self._or_key = openrouter_api_key.strip()
        self._ds_client: Any = None
        self._or_client: Any = None

        if not self._ds_key and not self._or_key:
            logger.debug("DeepSeekBrain: no API keys — provider disabled")
            return

        try:
            import openai

            if self._ds_key:
                self._ds_client = openai.AsyncOpenAI(
                    base_url=_DS_BASE_URL,
                    api_key=self._ds_key,
                    timeout=_TIMEOUT,
                )
                logger.info("DeepSeekBrain initialized via direct API (model=%s)", _DS_MODEL)

            if self._or_key:
                self._or_client = openai.AsyncOpenAI(
                    base_url=_OR_BASE_URL,
                    api_key=self._or_key,
                    timeout=_TIMEOUT,
                )
                if not self._ds_client:
                    logger.info("DeepSeekBrain initialized via OpenRouter (model=%s)", _OR_MODEL)

        except ImportError:
            logger.warning("DeepSeekBrain: openai package not installed — provider disabled")

    async def analyze(self, intelligence_package: dict[str, Any]) -> BrainAnalysis | None:
        """Run DeepSeek analysis. Returns None silently if not configured."""
        if self._ds_client is None and self._or_client is None:
            return None

        system_prompt = self._build_system_prompt()
        user_prompt = self._build_intelligence_prompt(intelligence_package)

        if self._ds_client is not None:
            result = await self._call_deepseek(system_prompt, user_prompt)
            if result is not None:
                return result
            logger.warning("DeepSeekBrain: direct API failed, trying OpenRouter fallback")

        if self._or_client is not None:
            return await self._call_openrouter(system_prompt, user_prompt)

        return None

    async def _call_deepseek(self, system_prompt: str, user_prompt: str) -> BrainAnalysis | None:
        try:
            response = await self._ds_client.chat.completions.create(
                model=_DS_MODEL,
                max_tokens=_MAX_TOKENS,
                temperature=_TEMPERATURE,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except Exception as exc:
            logger.warning("DeepSeekBrain direct API error: %s", exc)
            return None
        return self._extract_and_parse(response)

    async def _call_openrouter(self, system_prompt: str, user_prompt: str) -> BrainAnalysis | None:
        try:
            response = await self._or_client.chat.completions.create(
                model=_OR_MODEL,
                max_tokens=_MAX_TOKENS,
                temperature=_TEMPERATURE,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                extra_headers=_OR_HEADERS,
            )
        except Exception as exc:
            logger.warning("DeepSeekBrain OpenRouter error: %s", exc)
            return None
        return self._extract_and_parse(response)

    def _extract_and_parse(self, response: Any) -> BrainAnalysis | None:
        try:
            raw = response.choices[0].message.content
        except (AttributeError, IndexError) as exc:
            logger.warning("DeepSeekBrain: unexpected response shape: %s", exc)
            return None

        if not raw:
            logger.warning("DeepSeekBrain: empty response from API")
            return None

        result = self._parse_response(raw)
        if result is None:
            logger.warning("DeepSeekBrain: failed to parse response (len=%d)", len(raw))
        else:
            sig = f"{result.signal.action}@{result.signal.confidence:.2f}" if result.signal else "no-signal"
            logger.info("DeepSeekBrain: %s", sig)

        return result
