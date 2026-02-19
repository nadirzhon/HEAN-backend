"""LLM Providers for HEAN Sovereign Brain.

Available providers (all return BrainAnalysis | None):
- GroqBrain       — Fast analysis (Llama-3.3-70B via Groq, free tier)
- DeepSeekBrain   — Deep reasoning (DeepSeek-R1, $0.55/M tokens or via OpenRouter)
- OllamaBrain     — Local analysis (no API cost, requires Ollama server)

All providers share BaseLLMProvider which enforces a common interface.
"""

from .base import BaseLLMProvider
from .deepseek_brain import DeepSeekBrain
from .groq_brain import GroqBrain
from .ollama_brain import OllamaBrain

__all__ = [
    "BaseLLMProvider",
    "DeepSeekBrain",
    "GroqBrain",
    "OllamaBrain",
]
