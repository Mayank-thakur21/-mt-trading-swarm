"""SentimentAgent (LLM path, lexicon fallback, abstention) and the lexicon scorer."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from trading_swarm.agents import SentimentAgent
from trading_swarm.agents.lexicon import (
    NEGATION_FACTOR,
    lemmatize,
    score_headline,
    score_headlines,
    tokenize,
)
from trading_swarm.config import Settings
from trading_swarm.data import MarketDataError, SyntheticProvider
from trading_swarm.llm import NullLLMClient
from trading_swarm.models import HeadlineSentiment, NewsItem, SentimentJudgement


def _agent(provider: Any, llm: Any, settings: Settings) -> SentimentAgent:
    return SentimentAgent(provider, llm, settings)


# ----------------------------------------------------------------------------- LLM path


def test_llm_judgement_is_used(
    provider: SyntheticProvider, fake_llm: Any, settings: Settings
) -> None:
    signal = _agent(provider, fake_llm, settings).analyze("BULL")
    assert signal.agent == "sentiment"
    assert signal.signal == pytest.approx(0.6)
    assert signal.confidence == pytest.approx(0.8)
    assert signal.data["method"] == "fake:fake-model"
    assert signal.data["summary"].startswith("Headlines point to")
    assert signal.rationale == signal.data["summary"]
    # The LLM saw the provider's headlines for the right symbol.
    symbol, headlines = fake_llm.calls[0]
    assert symbol == "BULL"
    assert [h.title for h in headlines] == [h.title for h in provider.news("BULL", 10)]
    assert len(signal.data["headlines"]) == len(headlines)
    first = signal.data["headlines"][0]
    assert first["sentiment"] == "bullish" and first["relevance"] == 1.0


def test_claude_method_label_is_the_model_id(
    provider: SyntheticProvider,
    fake_llm_factory: Any,
    make_judgement: Callable[..., SentimentJudgement],
    settings: Settings,
) -> None:
    llm = fake_llm_factory(make_judgement(), name="anthropic", model="claude-opus-5-5")
    signal = _agent(provider, llm, settings).analyze("BULL")
    assert signal.data["method"] == "claude-opus-5-5"


def test_signal_is_relevance_weighted(
    provider: SyntheticProvider, fake_llm_factory: Any, settings: Settings
) -> None:
    judgement = SentimentJudgement(
        items=[
            HeadlineSentiment(index=0, sentiment="bullish", score=0.9, relevance=1.0),
            HeadlineSentiment(index=1, sentiment="bearish", score=-0.6, relevance=0.25),
            HeadlineSentiment(index=2, sentiment="neutral", score=0.0, relevance=0.0),
        ],
        overall_score=0.1,
        confidence=0.7,
        summary="Mostly positive.",
    )
    signal = _agent(provider, fake_llm_factory(judgement), settings).analyze("BULL")
    expected = (0.9 * 1.0 - 0.6 * 0.25) / 1.25
    assert signal.signal == pytest.approx(expected, abs=1e-4)
    assert signal.data["overall_score"] == pytest.approx(0.1)
    assert signal.confidence == pytest.approx(0.7)


def test_overall_score_used_when_no_item_is_relevant(
    provider: SyntheticProvider,
    fake_llm_factory: Any,
    make_judgement: Callable[..., SentimentJudgement],
    settings: Settings,
) -> None:
    judgement = make_judgement(score=-0.4, confidence=0.5, relevance=0.0)
    signal = _agent(provider, fake_llm_factory(judgement), settings).analyze("BEAR")
    assert signal.signal == pytest.approx(-0.4)
    assert signal.data["relevance_weighted_score"] is None


def test_hallucinated_indices_are_ignored(
    provider: SyntheticProvider, fake_llm_factory: Any, settings: Settings
) -> None:
    judgement = SentimentJudgement(
        items=[
            HeadlineSentiment(index=0, sentiment="bullish", score=0.5, relevance=1.0),
            HeadlineSentiment(index=0, sentiment="bearish", score=-1.0, relevance=1.0),  # dup
            HeadlineSentiment(index=99, sentiment="bearish", score=-1.0, relevance=1.0),  # bogus
        ],
        overall_score=-0.5,
        confidence=0.9,
    )
    signal = _agent(provider, fake_llm_factory(judgement), settings).analyze("BULL")
    assert signal.signal == pytest.approx(0.5)
    assert signal.data["summary"]  # default summary when the model gave none


def test_judgement_with_only_bogus_indices_falls_back_to_lexicon(
    provider: SyntheticProvider, fake_llm_factory: Any, settings: Settings
) -> None:
    judgement = SentimentJudgement(
        items=[HeadlineSentiment(index=42, sentiment="bearish", score=-1.0, relevance=1.0)],
        overall_score=-1.0,
        confidence=0.9,
    )
    signal = _agent(provider, fake_llm_factory(judgement), settings).analyze("BULL")
    assert signal.data["method"] == "lexicon"
    assert signal.signal > 0


# ----------------------------------------------------------------------------- fallbacks


def test_llm_none_falls_back_to_lexicon(
    provider: SyntheticProvider, fake_llm_factory: Any, settings: Settings
) -> None:
    llm = fake_llm_factory(None, name="anthropic", model="claude-opus-5-5")
    signal = _agent(provider, llm, settings).analyze("BULL")
    assert len(llm.calls) == 1
    assert signal.data["method"] == "lexicon"
    assert signal.data["llm_fallback_from"] == "claude-opus-5-5"
    assert signal.signal > 0.5
    assert 0 < signal.confidence <= SentimentAgent.LEXICON_MAX_CONFIDENCE


def test_disabled_llm_is_not_called(provider: SyntheticProvider, settings: Settings) -> None:
    class Spy(NullLLMClient):
        called = False

        def judge_sentiment(self, symbol: str, headlines: Any) -> None:
            Spy.called = True

    signal = _agent(provider, Spy(), settings).analyze("BEAR")
    assert not Spy.called
    assert signal.data["method"] == "lexicon"
    assert "llm_fallback_from" not in signal.data
    assert signal.signal < -0.5
    assert signal.confidence == pytest.approx(SentimentAgent.LEXICON_MAX_CONFIDENCE)


def test_raising_llm_client_falls_back(provider: SyntheticProvider, settings: Settings) -> None:
    class Broken:
        name = "openai"
        model = "gpt-5-mini"

        def judge_sentiment(self, symbol: str, headlines: Any) -> None:
            raise RuntimeError("contract violation")

    signal = _agent(provider, Broken(), settings).analyze("BULL")
    assert signal.data["method"] == "lexicon"
    assert signal.data["llm_fallback_from"] == "openai:gpt-5-mini"


def test_lexicon_on_neutral_news_abstains(provider: SyntheticProvider, settings: Settings) -> None:
    signal = _agent(provider, NullLLMClient(), settings).analyze("FLAT")
    assert signal.signal == 0.0
    assert signal.confidence == 0.0
    assert signal.data["coverage"] == 0.0
    assert signal.data["headline_count"] == 3


def test_lexicon_mixed_news_lowers_confidence(
    provider: SyntheticProvider, settings: Settings
) -> None:
    provider.set_news("MIX", ["{symbol} beats estimates", "{symbol} faces a fraud lawsuit"])
    signal = _agent(provider, NullLLMClient(), settings).analyze("MIX")
    assert signal.confidence < SentimentAgent.LEXICON_MAX_CONFIDENCE / 2
    per_headline = signal.data["headlines"]
    assert per_headline[0]["sentiment"] == "bullish"
    assert per_headline[1]["sentiment"] == "bearish"
    assert per_headline[0]["terms"] == ["beat estimate +2.0"]


def test_no_headlines_is_neutral_and_skips_llm(
    provider: SyntheticProvider, fake_llm: Any, settings: Settings
) -> None:
    provider.set_news("BULL", [])
    signal = _agent(provider, fake_llm, settings).analyze("BULL")
    assert (signal.signal, signal.confidence) == (0.0, 0.0)
    assert signal.rationale == "no recent headlines"
    assert fake_llm.calls == []


def test_news_outage_is_neutral(
    provider: SyntheticProvider, fake_llm: Any, settings: Settings
) -> None:
    # The synthetic "DOWN" symbol's news feed is unavailable → [] per the provider contract.
    signal = _agent(provider, fake_llm, settings).analyze("DOWN")
    assert signal.confidence == 0.0
    assert fake_llm.calls == []


def test_provider_raising_market_data_error_is_neutral(fake_llm: Any, settings: Settings) -> None:
    class Failing:
        def news(self, symbol: str, limit: int = 10) -> list[NewsItem]:
            raise MarketDataError("news endpoint down")

    signal = _agent(Failing(), fake_llm, settings).analyze("AAPL")
    assert signal.confidence == 0.0


def test_at_most_ten_headlines(fake_llm: Any, settings: Settings) -> None:
    class Many:
        def news(self, symbol: str, limit: int = 10) -> list[NewsItem]:
            return [NewsItem(title=f"headline {i}") for i in range(25)]

    signal = _agent(Many(), fake_llm, settings).analyze("AAPL")
    assert len(fake_llm.calls[0][1]) == SentimentAgent.MAX_HEADLINES
    assert signal.data["headline_count"] == SentimentAgent.MAX_HEADLINES


def test_history_argument_is_accepted(provider: SyntheticProvider, settings: Settings) -> None:
    agent = _agent(provider, NullLLMClient(), settings)
    assert agent.analyze("BULL", provider.history("BULL")) == agent.analyze("BULL")


# ----------------------------------------------------------------------------- lexicon scorer


@pytest.mark.parametrize(
    "headline",
    [
        "ACME beats earnings estimates and raises full-year guidance",
        "Analysts upgrade ACME to buy as revenue growth accelerates",
        "ACME shares rally to a record high",
        "ACME announces a share buyback",
        "ACME cuts costs, profit jumps",
    ],
)
def test_bullish_headlines(headline: str) -> None:
    assert score_headline(headline).score > 0.3


@pytest.mark.parametrize(
    "headline",
    [
        "ACME misses estimates and slashes guidance",
        "ACME shares plunge as regulators open a fraud investigation",
        "Analysts downgrade ACME to sell amid weak demand",
        "ACME warns of losses as debt costs rise",
        "ACME files for bankruptcy",
    ],
)
def test_bearish_headlines(headline: str) -> None:
    assert score_headline(headline).score < -0.3


@pytest.mark.parametrize(
    "headline",
    ["ACME schedules its quarterly earnings call", "ACME names a new chief financial officer", ""],
)
def test_neutral_headlines(headline: str) -> None:
    result = score_headline(headline)
    assert result.score == 0.0
    assert not result.has_signal
    assert result.label == "neutral"


def test_negation_flips_and_damps() -> None:
    plain = score_headline("ACME beats estimates")
    negated = score_headline("ACME did not beat estimates")
    assert plain.raw > 0 > negated.raw
    assert negated.raw == pytest.approx(plain.raw * NEGATION_FACTOR)
    assert negated.hits[0].negated
    assert negated.hits[0].describe().startswith("NOT ")


@pytest.mark.parametrize(
    ("headline", "sign"),
    [
        ("ACME didn't beat estimates", -1),  # n't contraction
        ("No growth expected at ACME", -1),
        ("ACME is not facing a lawsuit", 1),  # negated negative
        ("ACME never recovered", -1),
        ("ACME fails to win approval", -1),
    ],
)
def test_negation_variants(headline: str, sign: int) -> None:
    assert score_headline(headline).score * sign > 0


def test_miss_is_bearish_and_negates_following_positive() -> None:
    result = score_headline("ACME misses profit forecast")
    assert result.score < score_headline("ACME misses").score < 0
    assert [hit.negated for hit in result.hits] == [False, True]


def test_negation_does_not_cross_conjunctions_or_punctuation() -> None:
    mixed = score_headline("ACME misses estimates but raises guidance")
    assert all(not hit.negated for hit in mixed.hits)
    assert mixed.score == pytest.approx(0.0)
    assert all(not h.negated for h in score_headline("No surprise, ACME beats estimates").hits)


def test_phrases_override_their_words_and_allow_fillers() -> None:
    result = score_headline("ACME raises full-year guidance")
    assert [hit.term for hit in result.hits] == ["raise guidance"]
    assert [hit.term for hit in score_headline("ACME cuts guidance").hits] == ["cut guidance"]
    # Too many filler words: no phrase, individual words only.
    loose = score_headline("ACME raises prices on several products despite guidance")
    assert "raise guidance" not in [hit.term for hit in loose.hits]


def test_ticker_tokens_can_be_ignored() -> None:
    assert score_headline("LOW shares are flat", ignore=["LOW"]).hits == ()
    assert score_headline("LOW shares are flat").hits  # "low" is a (bearish) word otherwise


def test_whole_words_only() -> None:
    assert score_headline("Highway project at ACME").hits == ()


def test_tokenize_and_lemmatize() -> None:
    assert tokenize("ACME's shares DIDN'T rally") == ["acme", "shares", "did", "not", "rally"]
    assert lemmatize("rallies") == "rally"
    assert lemmatize("plunged") == "plunge"
    assert lemmatize("slashes") == "slash"
    assert lemmatize("cutting") == "cut"
    assert lemmatize("raising") == "raise"
    assert lemmatize("zebra") == "zebra"


def test_score_headlines_aggregate() -> None:
    result = score_headlines(["ACME beats estimates", "ACME schedules a call"])
    assert result.coverage == 0.5
    assert result.agreement == 1.0
    assert result.signal == pytest.approx(score_headline("ACME beats estimates").score / 2)
    empty = score_headlines([])
    assert (empty.signal, empty.coverage, empty.agreement) == (0.0, 0.0, 0.0)
