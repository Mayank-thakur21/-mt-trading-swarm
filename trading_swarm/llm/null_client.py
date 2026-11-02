"""The disabled LLM provider: always answers ``None`` so callers use their fallback."""

from __future__ import annotations

from collections.abc import Sequence

from trading_swarm.models import NewsItem, SentimentJudgement


class NullLLMClient:
    """:class:`~trading_swarm.llm.base.LLMClient` that never calls anything."""

    name = "none"
    model = ""

    def judge_sentiment(
        self, symbol: str, headlines: Sequence[NewsItem]
    ) -> SentimentJudgement | None:
        return None


__all__ = ["NullLLMClient"]
