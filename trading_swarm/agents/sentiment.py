"""News-sentiment agent: Claude (or another LLM) first, a finance lexicon as fallback.

Flow
----
1. Fetch up to :attr:`SentimentAgent.MAX_HEADLINES` recent headlines with
   ``provider.news(symbol, 10)``. No headlines (or the news feed is down) → the agent
   abstains: signal 0, confidence 0, so it neither moves the consensus nor dilutes it.
2. If an LLM is enabled (:func:`trading_swarm.llm.is_llm_enabled`), ask it for a
   structured :class:`~trading_swarm.models.SentimentJudgement`:

   * ``signal = Σ scoreᵢ·relevanceᵢ / Σ relevanceᵢ`` over the per-headline items
     (:meth:`~trading_swarm.models.SentimentJudgement.relevance_weighted_score`) — a
     headline that is only tangentially about the company counts less; falls back to
     the model's ``overall_score`` when no item carries any relevance;
   * ``confidence = judgement.confidence``.

   Items whose ``index`` does not refer to a supplied headline (a hallucinated or
   duplicated index) are discarded before weighting.
3. Otherwise — no LLM configured, or the LLM returned ``None`` (it never raises, see
   :mod:`trading_swarm.llm.base`) — score the headlines with the lexicon
   (:mod:`trading_swarm.agents.lexicon`). Keyword scoring is crude, so its confidence is
   ``0.4 × coverage × agreement`` and never exceeds :attr:`LEXICON_MAX_CONFIDENCE`
   (``coverage`` = share of headlines with any sentiment term, ``agreement`` =
   ``|Σ sᵢ| / Σ |sᵢ|``).

``data`` records the ``method`` (``"claude-opus-5-5"``, ``"openai:<model>"`` or
``"lexicon"``), the per-headline scores and a summary for the dashboard.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any, Final

import pandas as pd

from trading_swarm.agents.lexicon import LexiconResult, score_headlines
from trading_swarm.config import Settings
from trading_swarm.data.base import MarketDataError, MarketDataProvider
from trading_swarm.llm.base import LLMClient, is_llm_enabled, llm_method_label
from trading_swarm.models import AgentSignal, HeadlineSentiment, NewsItem, SentimentJudgement

logger = logging.getLogger(__name__)

LEXICON_METHOD: Final = "lexicon"


def _published(item: NewsItem) -> str | None:
    return item.published_at.isoformat() if item.published_at is not None else None


class SentimentAgent:
    """Votes from recent headlines (LLM first, lexicon fallback)."""

    name = "sentiment"
    #: Lexicon-based votes are never more confident than this.
    LEXICON_MAX_CONFIDENCE = 0.4
    #: Headlines requested from the provider.
    MAX_HEADLINES = 10

    def __init__(self, provider: MarketDataProvider, llm: LLMClient, settings: Settings) -> None:
        self.provider = provider
        self.llm = llm
        self.settings = settings

    # ------------------------------------------------------------------ public API
    def analyze(self, symbol: str, history: pd.DataFrame | None = None) -> AgentSignal:
        """Sentiment vote for ``symbol`` (``history`` is accepted for protocol symmetry)."""
        headlines = self._headlines(symbol)
        if not headlines:
            return AgentSignal.neutral(
                self.name, "no recent headlines", method="none", headline_count=0, headlines=[]
            )

        llm_label: str | None = None
        if is_llm_enabled(self.llm):
            llm_label = llm_method_label(self.llm)
            judgement = self._judge(symbol, headlines)
            if judgement is not None:
                vote = self._from_judgement(symbol, headlines, judgement, llm_label)
                if vote is not None:
                    return vote
            logger.info("sentiment %s: %s unavailable, using the lexicon", symbol, llm_label)
        return self._from_lexicon(symbol, headlines, llm_fallback_from=llm_label)

    # ------------------------------------------------------------------ inputs
    def _headlines(self, symbol: str) -> list[NewsItem]:
        try:
            items = self.provider.news(symbol, self.MAX_HEADLINES)
        except MarketDataError as exc:
            # The contract says news() returns [] on failure; be defensive anyway.
            logger.warning("sentiment %s: news unavailable (%s)", symbol, exc)
            return []
        return list(items)[: self.MAX_HEADLINES]

    def _judge(self, symbol: str, headlines: Sequence[NewsItem]) -> SentimentJudgement | None:
        try:
            return self.llm.judge_sentiment(symbol, headlines)
        except Exception:
            # LLM clients must never raise; a broken client must not break a decision.
            logger.exception("sentiment %s: LLM client raised; falling back to lexicon", symbol)
            return None

    # ------------------------------------------------------------------ LLM path
    def _from_judgement(
        self,
        symbol: str,
        headlines: Sequence[NewsItem],
        judgement: SentimentJudgement,
        method: str,
    ) -> AgentSignal | None:
        """Vote from a structured LLM judgement (``None`` if it scored no real headline)."""
        valid: dict[int, HeadlineSentiment] = {}
        for item in judgement.items:
            if 0 <= item.index < len(headlines) and item.index not in valid:
                valid[item.index] = item
        dropped = len(judgement.items) - len(valid)
        if dropped:
            logger.warning(
                "sentiment %s: ignored %d LLM item(s) with invalid/duplicate indices",
                symbol,
                dropped,
            )
        if not valid and judgement.items:
            logger.warning("sentiment %s: LLM judgement referenced no supplied headline", symbol)
            return None

        cleaned = judgement.model_copy(update={"items": list(valid.values())})
        weighted = cleaned.relevance_weighted_score()
        signal = weighted if weighted is not None else judgement.overall_score
        scored = [
            {
                "index": i,
                "title": item.title,
                "publisher": item.publisher,
                "published_at": _published(item),
                "sentiment": valid[i].sentiment if i in valid else None,
                "score": round(valid[i].score, 4) if i in valid else None,
                "relevance": round(valid[i].relevance, 4) if i in valid else None,
            }
            for i, item in enumerate(headlines)
        ]
        summary = judgement.summary or self._default_summary(signal, len(headlines))
        return AgentSignal(
            agent=self.name,
            signal=round(signal, 4),
            confidence=round(judgement.confidence, 4),
            rationale=summary,
            data={
                "method": method,
                "headline_count": len(headlines),
                "overall_score": round(judgement.overall_score, 4),
                "relevance_weighted_score": None if weighted is None else round(weighted, 4),
                "summary": summary,
                "headlines": scored,
            },
        )

    # ------------------------------------------------------------------ lexicon path
    def _from_lexicon(
        self, symbol: str, headlines: Sequence[NewsItem], *, llm_fallback_from: str | None
    ) -> AgentSignal:
        result = score_headlines((item.title for item in headlines), ignore=[symbol])
        confidence = min(
            self.LEXICON_MAX_CONFIDENCE,
            self.LEXICON_MAX_CONFIDENCE * result.coverage * result.agreement,
        )
        summary = self._lexicon_summary(result)
        data: dict[str, Any] = {
            "method": LEXICON_METHOD,
            "headline_count": len(headlines),
            "coverage": round(result.coverage, 4),
            "agreement": round(result.agreement, 4),
            "summary": summary,
            "headlines": [
                {
                    "index": i,
                    "title": item.title,
                    "publisher": item.publisher,
                    "published_at": _published(item),
                    "sentiment": score.label,
                    "score": round(score.score, 4),
                    "terms": [hit.describe() for hit in score.hits],
                }
                for i, (item, score) in enumerate(zip(headlines, result.headlines, strict=True))
            ],
        }
        if llm_fallback_from is not None:
            data["llm_fallback_from"] = llm_fallback_from
        return AgentSignal(
            agent=self.name,
            signal=round(result.signal, 4),
            confidence=round(confidence, 4),
            rationale=summary,
            data=data,
        )

    @staticmethod
    def _lexicon_summary(result: LexiconResult) -> str:
        bullish = sum(1 for s in result.headlines if s.label == "bullish")
        bearish = sum(1 for s in result.headlines if s.label == "bearish")
        neutral = len(result.headlines) - bullish - bearish
        text = f"{bullish} bullish, {bearish} bearish, {neutral} neutral of {len(result.headlines)}"
        # The single most opinionated headline (first one wins ties: deterministic).
        strongest = max(
            (s for s in result.headlines if s.has_signal), key=lambda s: abs(s.score), default=None
        )
        if strongest is not None:
            text += f'; strongest: "{strongest.text}" ({strongest.score:+.2f})'
        return text

    @staticmethod
    def _default_summary(signal: float, count: int) -> str:
        tone = "bullish" if signal > 0.1 else "bearish" if signal < -0.1 else "neutral"
        return f"{tone} tone across {count} headline(s)"


__all__ = ["LEXICON_METHOD", "SentimentAgent"]
