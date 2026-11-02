"""LLM client contract for headline sentiment.

Contract: :meth:`LLMClient.judge_sentiment` **never raises** to its caller. Any failure —
missing credentials, network errors, rate limits, refusals, truncated or malformed
output — is logged (with the provider's request id when available) and reported as
``None``, upon which the sentiment agent falls back to its lexicon scorer. A trading
decision must never fail because a language model was unavailable.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from trading_swarm.models import HeadlineSentiment, NewsItem, SentimentJudgement


class LLMError(Exception):
    """Internal error raised inside a client before being converted into ``None``."""


@runtime_checkable
class LLMClient(Protocol):
    """A provider that scores headlines for one ticker."""

    #: Provider id: ``"anthropic"``, ``"openai"`` or ``"none"``.
    name: str
    #: Model id used for requests (empty string when disabled).
    model: str

    def judge_sentiment(
        self, symbol: str, headlines: Sequence[NewsItem]
    ) -> SentimentJudgement | None:
        """Structured sentiment for ``headlines`` (``None`` on any failure / when disabled)."""
        ...


def is_llm_enabled(client: LLMClient | None) -> bool:
    """``False`` for ``None`` or the null provider."""
    return client is not None and client.name != "none"


def llm_method_label(client: LLMClient) -> str:
    """Label recorded in ``AgentSignal.data["method"]``.

    Claude is the default provider, so its label is just the model id
    (``"claude-opus-5-5"``); other providers are prefixed (``"openai:gpt-5-mini"``).
    """
    if client.name == "anthropic":
        return client.model
    return f"{client.name}:{client.model}" if client.model else client.name


__all__ = [
    "HeadlineSentiment",
    "LLMClient",
    "LLMError",
    "SentimentJudgement",
    "is_llm_enabled",
    "llm_method_label",
]
