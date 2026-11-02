"""LLM providers for headline sentiment (Claude by default, OpenAI optional, or disabled)."""

from __future__ import annotations

import logging

from trading_swarm.config import Settings
from trading_swarm.llm.base import (
    LLMClient,
    LLMError,
    SentimentJudgement,
    is_llm_enabled,
    llm_method_label,
)
from trading_swarm.llm.null_client import NullLLMClient

logger = logging.getLogger(__name__)


def get_llm_client(settings: Settings) -> LLMClient:
    """Build the client selected by ``LLM_PROVIDER``.

    A provider without credentials (or whose SDK client cannot be constructed) degrades to
    :class:`NullLLMClient` with a warning, so sentiment falls back to the lexicon scorer
    instead of the application failing to start.
    """
    provider = settings.llm_provider
    if provider == "none":
        logger.info("LLM sentiment disabled (LLM_PROVIDER=none); using the lexicon scorer")
        return NullLLMClient()

    if provider == "anthropic":
        if not settings.anthropic_credentials_available:
            logger.warning(
                "LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY is empty (set it, or set "
                "ANTHROPIC_USE_SDK_DEFAULT_AUTH=true to use the SDK's credential chain); "
                "falling back to lexicon sentiment"
            )
            return NullLLMClient()
        try:
            from trading_swarm.llm.anthropic_client import AnthropicSentimentClient

            client: LLMClient = AnthropicSentimentClient(settings)
        except Exception as exc:
            logger.warning("could not initialise the Claude client (%s); using lexicon", exc)
            return NullLLMClient()
        logger.info(
            "LLM sentiment via Claude (%s, effort=%s)",
            settings.anthropic_model,
            settings.anthropic_effort,
        )
        return client

    if provider == "openai":
        if not settings.openai_api_key:
            logger.warning(
                "LLM_PROVIDER=openai but OPENAI_API_KEY is empty; falling back to lexicon sentiment"
            )
            return NullLLMClient()
        try:
            from trading_swarm.llm.openai_client import OpenAISentimentClient

            client = OpenAISentimentClient(settings)
        except Exception as exc:
            logger.warning("could not initialise the OpenAI client (%s); using lexicon", exc)
            return NullLLMClient()
        logger.info("LLM sentiment via OpenAI (%s)", settings.openai_model)
        return client

    logger.warning("unknown LLM_PROVIDER %r; using lexicon sentiment", provider)  # pragma: no cover
    return NullLLMClient()  # pragma: no cover


__all__ = [
    "LLMClient",
    "LLMError",
    "NullLLMClient",
    "SentimentJudgement",
    "get_llm_client",
    "is_llm_enabled",
    "llm_method_label",
]
