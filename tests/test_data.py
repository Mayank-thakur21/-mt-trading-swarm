"""Market-data layer: symbol validation, normalisation, synthetic provider, Yahoo provider
(with a fake ``yf.Ticker`` — no network), TTL cache and provider factory."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
import pytest

from trading_swarm.config import Settings
from trading_swarm.data import (
    InvalidSymbolError,
    MarketDataError,
    MarketDataProvider,
    NoDataError,
    ProviderUnavailableError,
    SyntheticProvider,
    YahooFinanceProvider,
    daily_change_pct,
    get_market_data_provider,
    validate_symbol,
)
from trading_swarm.data import yahoo as yahoo_module
from trading_swarm.data.base import OHLCV_COLUMNS, apply_period, normalize_ohlcv, slice_range
from trading_swarm.data.cache import TTLCache
from trading_swarm.data.yahoo import parse_news_item
from trading_swarm.models import NewsItem

TEST_END_DATE = "2026-09-14"  # matches the ``synthetic_provider`` fixture in conftest.py

# ----------------------------------------------------------------------------- symbols


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("aapl", "AAPL"),
        (" msft ", "MSFT"),
        ("BRK-B", "BRK-B"),
        ("^gspc", "^GSPC"),
        ("eurusd=x", "EURUSD=X"),
        ("RDS.A", "RDS.A"),
    ],
)
def test_validate_symbol_normalises(raw: str, expected: str) -> None:
    assert validate_symbol(raw) == expected


@pytest.mark.parametrize(
    "raw", ["", "   ", "AAPL;DROP", "A B", "TOO-LONG-SYMBOL-X", "$AAPL", None, 42]
)
def test_validate_symbol_rejects(raw: Any) -> None:
    with pytest.raises(InvalidSymbolError):
        validate_symbol(raw)


def test_invalid_symbol_is_a_market_data_and_value_error() -> None:
    with pytest.raises(MarketDataError):
        validate_symbol("bad symbol")
    with pytest.raises(ValueError):
        validate_symbol("bad symbol")


# ----------------------------------------------------------------------------- normalisation


def _assert_canonical(df: pd.DataFrame) -> None:
    assert list(df.columns) == list(OHLCV_COLUMNS)
    assert isinstance(df.index, pd.DatetimeIndex)
    assert str(df.index.tz) == "UTC"
    assert df.index.is_monotonic_increasing and df.index.is_unique
    assert all(pd.api.types.is_float_dtype(dtype) for dtype in df.dtypes)
    assert (df["Close"] > 0).all()
    assert (df["High"] >= df[["Open", "Close"]].max(axis=1) - 1e-9).all()
    assert (df["Low"] <= df[["Open", "Close"]].min(axis=1) + 1e-9).all()


def test_normalize_handles_yfinance_quirks() -> None:
    index = pd.DatetimeIndex(
        [
            "2024-03-01 00:00",
            "2024-03-04 00:00",
            "2024-03-04 00:00",
            "2024-03-05 00:00",
            "2024-03-06 00:00",
        ],
        tz="America/New_York",
    )
    raw = pd.DataFrame(
        {
            ("Close", "AAPL"): [10.0, 11.0, 11.5, np.nan, 12.0],
            ("open", "AAPL"): [9.5, 10.5, 10.6, 11.0, 11.8],
            ("HIGH", "AAPL"): [10.5, 11.2, 11.4, 11.3, 11.9],  # last high < close: repaired
            ("Low", "AAPL"): [9.0, 10.0, 10.1, 10.8, 11.5],
            ("Volume", "AAPL"): [100, 200, 300, 400, np.nan],
        },
        index=index,
    )
    df = normalize_ohlcv(raw, symbol="AAPL")
    _assert_canonical(df)
    # Local trading dates stamped at midnight UTC, duplicate kept last, NaN close dropped.
    assert [ts.isoformat() for ts in df.index] == [
        "2024-03-01T00:00:00+00:00",
        "2024-03-04T00:00:00+00:00",
        "2024-03-06T00:00:00+00:00",
    ]
    assert df.loc["2024-03-04", "Close"].item() == 11.5
    assert df.loc["2024-03-06", "High"].item() == 12.0
    assert df.loc["2024-03-06", "Volume"].item() == 0.0


def test_normalize_adds_missing_volume_and_rejects_bad_frames() -> None:
    frame = pd.DataFrame(
        {"Open": [1.0], "High": [1.0], "Low": [1.0], "Close": [1.0]},
        index=pd.DatetimeIndex(["2024-01-02"]),
    )
    assert normalize_ohlcv(frame, symbol="X")["Volume"].tolist() == [0.0]
    with pytest.raises(NoDataError):
        normalize_ohlcv(pd.DataFrame(), symbol="X")
    with pytest.raises(NoDataError):
        normalize_ohlcv(None, symbol="X")
    with pytest.raises(MarketDataError, match="missing columns"):
        normalize_ohlcv(frame.drop(columns=["Close"]), symbol="X")
    with pytest.raises(NoDataError):
        normalize_ohlcv(frame.assign(Close=[-1.0]), symbol="X")


def test_intraday_intervals_keep_their_times() -> None:
    frame = pd.DataFrame(
        {"Open": [1.0, 1.1], "High": [1.2, 1.2], "Low": [0.9, 1.0], "Close": [1.1, 1.15]},
        index=pd.DatetimeIndex(["2024-01-02 09:30", "2024-01-02 09:35"], tz="America/New_York"),
    )
    df = normalize_ohlcv(frame, symbol="X", interval="5m")
    assert df.index[0] == pd.Timestamp("2024-01-02 14:30", tz="UTC")


def test_daily_change_pct() -> None:
    frame = pd.DataFrame({"Close": [100.0, 105.0]})
    assert daily_change_pct(frame) == pytest.approx(5.0)
    assert daily_change_pct(frame.iloc[:1]) is None


# ----------------------------------------------------------------------------- synthetic


def test_synthetic_provider_satisfies_protocol(synthetic_provider: SyntheticProvider) -> None:
    assert isinstance(synthetic_provider, MarketDataProvider)
    assert synthetic_provider.name == "synthetic"


def test_synthetic_history_shape(synthetic_provider: SyntheticProvider) -> None:
    df = synthetic_provider.history("AAPL")
    _assert_canonical(df)
    assert df.index[-1] == pd.Timestamp(TEST_END_DATE, tz="UTC")
    assert (df.index == df.index.normalize()).all()
    assert (df.index.day_of_week < 5).all()
    assert 248 <= len(df) <= 262


def test_synthetic_is_deterministic() -> None:
    a = SyntheticProvider(seed=7, end_date=TEST_END_DATE)
    b = SyntheticProvider(seed=7, end_date=TEST_END_DATE)
    pd.testing.assert_frame_equal(a.history("MSFT"), b.history("MSFT"))
    c = SyntheticProvider(seed=8, end_date=TEST_END_DATE)
    assert not a.history("MSFT")["Close"].equals(c.history("MSFT")["Close"])


def test_synthetic_symbols_are_independent_of_request_order() -> None:
    first = SyntheticProvider(seed=1, end_date=TEST_END_DATE)
    first.history("AAA")
    after = first.history("BBB")
    alone = SyntheticProvider(seed=1, end_date=TEST_END_DATE).history("BBB")
    pd.testing.assert_frame_equal(after, alone)


def test_extending_the_calendar_keeps_earlier_bars() -> None:
    short = SyntheticProvider(seed=3, end_date=TEST_END_DATE).history("XYZ", period="max")
    long = SyntheticProvider(seed=3, end_date="2027-06-30").history("XYZ", period="max")
    pd.testing.assert_frame_equal(short, long.loc[short.index], check_freq=False)


def test_scenarios_trend_as_named(synthetic_provider: SyntheticProvider) -> None:
    def one_year_return(symbol: str) -> float:
        close = synthetic_provider.history(symbol)["Close"]
        return float(close.iloc[-1] / close.iloc[0] - 1)

    assert one_year_return("BULL") > 0.3
    assert one_year_return("BEAR") < -0.3
    assert abs(one_year_return("FLAT")) < 0.1
    flat = synthetic_provider.history("FLAT")["Close"]
    assert flat.max() / flat.min() < 1.25  # range-bound


def test_scenario_anchor_sets_price_level(synthetic_provider: SyntheticProvider) -> None:
    bull = synthetic_provider.history("BULL", start="2026-01-02", end="2026-01-02")
    assert bull["Close"].item() == pytest.approx(100.0)


def test_drift_regimes_and_overrides() -> None:
    provider = SyntheticProvider(
        seed=5,
        drift={"CYCL": [("2021-01-01", 0.9), ("2024-01-01", -0.9)]},
        vol={"CYCL": 0.05},
        start_price={"CYCL": 50.0},
        end_date="2025-12-31",
    )
    close = provider.history("CYCL", period="max")["Close"]
    assert close.iloc[0] == pytest.approx(50.0)
    assert close.loc[:"2023-12-29"].iloc[-1] > close.iloc[0]
    assert close.iloc[-1] < close.loc[:"2023-12-29"].iloc[-1]
    assert provider.scenario("cycl").vol == 0.05


def test_history_ranges_are_inclusive(synthetic_provider: SyntheticProvider) -> None:
    df = synthetic_provider.history("AAPL", start="2024-01-02", end="2024-01-31")
    assert df.index[0] == pd.Timestamp("2024-01-02", tz="UTC")
    assert df.index[-1] == pd.Timestamp("2024-01-31", tz="UTC")
    trailing = synthetic_provider.history("AAPL", end="2024-06-28", period="1mo")
    assert trailing.index[-1] == pd.Timestamp("2024-06-28", tz="UTC")
    assert 19 <= len(trailing) <= 23


@pytest.mark.parametrize(("period", "bars"), [("1d", 1), ("5d", 5)])
def test_short_periods_count_bars(
    synthetic_provider: SyntheticProvider, period: str, bars: int
) -> None:
    assert len(synthetic_provider.history("AAPL", period=period)) == bars


def test_ytd_and_max_periods(synthetic_provider: SyntheticProvider) -> None:
    ytd = synthetic_provider.history("AAPL", period="ytd")
    assert ytd.index[0].year == pd.Timestamp(TEST_END_DATE).year
    assert ytd.index[0].month == 1
    full = synthetic_provider.history("AAPL", period="max")
    assert full.index[0] == pd.Timestamp("2021-01-04", tz="UTC")


def test_weekly_and_monthly_resampling(synthetic_provider: SyntheticProvider) -> None:
    daily = synthetic_provider.history("AAPL", start="2024-01-01", end="2024-03-29")
    weekly = synthetic_provider.history(
        "AAPL", start="2024-01-01", end="2024-03-29", interval="1wk"
    )
    monthly = synthetic_provider.history(
        "AAPL", start="2024-01-01", end="2024-03-29", interval="1mo"
    )
    _assert_canonical(weekly)
    assert len(weekly) == 13 and len(monthly) == 3
    assert monthly["Volume"].sum() == pytest.approx(daily["Volume"].sum())
    assert monthly["High"].iloc[0] == pytest.approx(daily.loc[:"2024-01-31", "High"].max())


def test_bad_period_and_interval(synthetic_provider: SyntheticProvider) -> None:
    with pytest.raises(MarketDataError, match="interval"):
        synthetic_provider.history("AAPL", interval="1m")
    with pytest.raises(MarketDataError, match="period"):
        synthetic_provider.history("AAPL", period="7y")
    with pytest.raises(NoDataError):
        synthetic_provider.history("AAPL", start="1999-01-01", end="1999-12-31")


def test_latest_price_is_last_close(synthetic_provider: SyntheticProvider) -> None:
    last_close = synthetic_provider.history("AAPL")["Close"].iloc[-1]
    assert synthetic_provider.latest_price("aapl") == last_close


def test_error_symbols(synthetic_provider: SyntheticProvider) -> None:
    with pytest.raises(NoDataError):
        synthetic_provider.history("NOPE")
    with pytest.raises(NoDataError):
        synthetic_provider.latest_price("NOPE")
    with pytest.raises(ProviderUnavailableError):
        synthetic_provider.history("DOWN")
    with pytest.raises(InvalidSymbolError):
        synthetic_provider.history("not valid")
    assert synthetic_provider.news("NOPE") == []
    assert synthetic_provider.news("DOWN") == []


def test_scenario_news(synthetic_provider: SyntheticProvider) -> None:
    news = synthetic_provider.news("BULL", limit=3)
    assert len(news) == 3
    assert all(isinstance(item, NewsItem) for item in news)
    assert "BULL" in news[0].title and "beats" in news[0].title
    dates = [item.published_at for item in news]
    assert all(d is not None and d.tzinfo is not None for d in dates)
    assert dates == sorted(dates, reverse=True)
    assert synthetic_provider.news("BULL", limit=0) == []


def test_generated_news_follow_the_trend() -> None:
    provider = SyntheticProvider(
        seed=2, drift={"UPPY": 3.0}, vol={"UPPY": 0.05}, end_date=TEST_END_DATE
    )
    titles = " ".join(item.title for item in provider.news("UPPY"))
    assert "UPPY" in titles
    assert any(
        word in titles for word in ("beats", "record", "upgrade", "buyback", "wins", "margins")
    )


def test_custom_news_and_history_overrides(
    synthetic_provider: SyntheticProvider, make_ohlcv: Callable[..., pd.DataFrame]
) -> None:
    custom = NewsItem(title="Custom headline", publisher="Unit Test")
    synthetic_provider.set_news("ACME", ["ACME {symbol} wins award", custom])
    news = synthetic_provider.news("ACME")
    assert news[0].title == "ACME ACME wins award"
    assert news[1] is custom

    frame = make_ohlcv([10.0, 11.0, 12.0]).rename(columns=str.lower)
    frame.index = frame.index.tz_localize(None)
    synthetic_provider.set_history("ACME", frame)
    df = synthetic_provider.history("ACME", period="max")
    _assert_canonical(df)
    assert df["Close"].tolist() == [10.0, 11.0, 12.0]
    assert synthetic_provider.latest_price("ACME") == 12.0


def test_scenario_validation() -> None:
    from trading_swarm.data.synthetic import Scenario

    with pytest.raises(ValueError):
        Scenario(vol=-1)
    with pytest.raises(ValueError):
        Scenario(start_price=0)
    with pytest.raises(ValueError):
        SyntheticProvider(origin="2025-01-01", end_date="2024-01-01")


def test_period_helpers() -> None:
    frame = pd.DataFrame(
        {"Close": range(400)},
        index=pd.bdate_range("2023-01-02", periods=400, tz="UTC"),
    )
    assert len(apply_period(frame, "max")) == 400
    assert apply_period(frame, "1y").index[0] > frame.index[-1] - pd.DateOffset(years=1)
    with pytest.raises(ValueError):
        apply_period(frame, "3w")
    sliced = slice_range(frame, start="2023-02-01", end="2023-02-28")
    assert sliced.index[0].day == 1 and sliced.index[-1].day == 28


# ----------------------------------------------------------------------------- TTL cache


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_ttl_cache_expiry_and_eviction() -> None:
    clock = _Clock()
    cache: TTLCache[int] = TTLCache(ttl_seconds=10, maxsize=2, clock=clock)
    cache.set("a", 1)
    assert cache.get("a") == 1
    clock.now = 10.0
    assert cache.get("a") is None  # expired

    cache.set("a", 1)
    cache.set("b", 2)
    cache.get("a")  # refresh LRU position
    cache.set("c", 3)  # evicts "b"
    assert cache.get("b") is None and cache.get("a") == 1 and len(cache) == 2

    calls: list[int] = []
    assert cache.get_or_set("d", lambda: calls.append(1) or 4) == 4
    assert cache.get_or_set("d", lambda: calls.append(1) or 5) == 4
    assert calls == [1]
    cache.clear()
    assert len(cache) == 0


def test_ttl_zero_disables_caching() -> None:
    cache: TTLCache[int] = TTLCache(ttl_seconds=0)
    cache.set("k", 1)
    assert cache.get("k") is None
    with pytest.raises(ValueError):
        TTLCache(ttl_seconds=-1)


# ----------------------------------------------------------------------------- yahoo


NEW_SHAPE_NEWS = {
    "id": "abc",
    "content": {
        "title": "  Apple beats   estimates ",
        "summary": "Strong iPhone demand.",
        "pubDate": "2024-05-02T20:30:00Z",
        "provider": {"displayName": "Reuters"},
        "canonicalUrl": {"url": "https://example.com/a"},
    },
}
OLD_SHAPE_NEWS = {
    "title": "Apple faces EU probe",
    "publisher": "Bloomberg",
    "link": "https://example.com/b",
    "providerPublishTime": 1714600000,
}


class FakeTicker:
    """Stands in for ``yf.Ticker`` (records calls; configurable failures)."""

    def __init__(
        self,
        symbol: str,
        *,
        frame: pd.DataFrame | None = None,
        failures: int = 0,
        error: BaseException | None = None,
        news: list[Any] | None = None,
        fast_price: Any = 187.5,
    ) -> None:
        self.symbol = symbol
        self._frame = frame
        self._failures = failures
        self._error = error
        self._news = news if news is not None else [NEW_SHAPE_NEWS, OLD_SHAPE_NEWS]
        self._fast_price = fast_price
        self.history_calls: list[dict[str, Any]] = []

    def history(self, **kwargs: Any) -> pd.DataFrame:
        self.history_calls.append(kwargs)
        if self._error is not None and self._failures != 0:
            self._failures -= 1
            raise self._error
        return self._frame if self._frame is not None else pd.DataFrame()

    @property
    def fast_info(self) -> dict[str, Any]:
        if isinstance(self._fast_price, BaseException):
            raise self._fast_price
        return {"lastPrice": self._fast_price}

    def get_news(self, count: int = 10) -> list[Any]:
        if isinstance(self._news, BaseException):
            raise self._news
        return self._news


def _yahoo_frame() -> pd.DataFrame:
    index = pd.DatetimeIndex(
        ["2024-05-01 00:00", "2024-05-02 00:00", "2024-05-03 00:00"], tz="America/New_York"
    )
    return pd.DataFrame(
        {
            "Open": [170.0, 171.0, 172.0],
            "High": [172.0, 173.0, 174.0],
            "Low": [169.0, 170.0, 171.0],
            "Close": [171.0, 172.0, 173.0],
            "Volume": [1e6, 2e6, 3e6],
            "Dividends": [0.0, 0.0, 0.0],
        },
        index=index,
    )


def _yahoo(ticker: FakeTicker, **kwargs: Any) -> tuple[YahooFinanceProvider, list[float]]:
    sleeps: list[float] = []
    provider = YahooFinanceProvider(
        ticker_factory=lambda symbol: ticker, sleep=sleeps.append, backoff_seconds=0.25, **kwargs
    )
    return provider, sleeps


def test_yahoo_history_normalises_and_caches() -> None:
    ticker = FakeTicker("AAPL", frame=_yahoo_frame())
    provider, _ = _yahoo(ticker)
    df = provider.history("aapl", period="6mo")
    _assert_canonical(df)
    assert df.index[0] == pd.Timestamp("2024-05-01", tz="UTC")
    provider.history("AAPL", period="6mo")
    assert len(ticker.history_calls) == 1  # second call served from the cache
    call = ticker.history_calls[0]
    assert call["period"] == "6mo" and call["auto_adjust"] is True and call["interval"] == "1d"


def test_yahoo_end_is_inclusive() -> None:
    ticker = FakeTicker("AAPL", frame=_yahoo_frame())
    provider, _ = _yahoo(ticker)
    provider.history("AAPL", start="2024-05-01", end="2024-05-03")
    call = ticker.history_calls[0]
    assert call["start"] == "2024-05-01"
    assert call["end"] == "2024-05-04"  # yfinance's end is exclusive


def test_yahoo_trailing_period_ending_at_end() -> None:
    ticker = FakeTicker("AAPL", frame=_yahoo_frame())
    provider, _ = _yahoo(ticker)
    df = provider.history("AAPL", end="2024-05-02", period="5d")
    assert df.index[-1] == pd.Timestamp("2024-05-02", tz="UTC")
    assert "start" in ticker.history_calls[0]


def test_yahoo_retries_transient_errors() -> None:
    ticker = FakeTicker("AAPL", frame=_yahoo_frame(), failures=2, error=TimeoutError("slow"))
    provider, sleeps = _yahoo(ticker, max_retries=2)
    assert len(provider.history("AAPL")) == 3
    assert len(ticker.history_calls) == 3
    assert sleeps == [0.25, 0.5]  # exponential back-off


def test_yahoo_gives_up_after_retries() -> None:
    ticker = FakeTicker("AAPL", failures=-1, error=ConnectionError("reset"))
    provider, sleeps = _yahoo(ticker, max_retries=1)
    with pytest.raises(ProviderUnavailableError, match="after 2 attempts"):
        provider.history("AAPL")
    assert len(ticker.history_calls) == 2 and sleeps == [0.25]


def test_yahoo_empty_frame_is_no_data_without_retry() -> None:
    ticker = FakeTicker("ZZZZ", frame=pd.DataFrame())
    provider, sleeps = _yahoo(ticker)
    with pytest.raises(NoDataError):
        provider.history("ZZZZ")
    assert len(ticker.history_calls) == 1 and sleeps == []


def test_yahoo_missing_ticker_exception_is_no_data() -> None:
    missing = getattr(yahoo_module.yf.exceptions, "YFTickerMissingError", None)
    if missing is None:  # pragma: no cover - older yfinance
        pytest.skip("yfinance has no YFTickerMissingError")
    error = missing("ZZZZ", "no timezone found")
    ticker = FakeTicker("ZZZZ", failures=-1, error=error)
    provider, sleeps = _yahoo(ticker)
    with pytest.raises(NoDataError):
        provider.history("ZZZZ")
    assert sleeps == []


def test_yahoo_rejects_unknown_period_before_calling() -> None:
    ticker = FakeTicker("AAPL", frame=_yahoo_frame())
    provider, _ = _yahoo(ticker)
    with pytest.raises(MarketDataError):
        provider.history("AAPL", period="7y")
    assert ticker.history_calls == []


def test_yahoo_latest_price_prefers_fast_info() -> None:
    provider, _ = _yahoo(FakeTicker("AAPL", frame=_yahoo_frame(), fast_price=187.5))
    assert provider.latest_price("AAPL") == 187.5


@pytest.mark.parametrize("fast_price", [RuntimeError("boom"), None, float("nan"), -1.0])
def test_yahoo_latest_price_falls_back_to_last_close(fast_price: Any) -> None:
    provider, _ = _yahoo(FakeTicker("AAPL", frame=_yahoo_frame(), fast_price=fast_price))
    assert provider.latest_price("AAPL") == 173.0


def test_yahoo_news_handles_both_shapes() -> None:
    duplicate = {"title": "Apple faces EU probe", "publisher": "Other"}
    junk = ["not a dict", {"title": "   "}, {"content": {"title": None}}]
    provider, _ = _yahoo(
        FakeTicker("AAPL", news=[OLD_SHAPE_NEWS, NEW_SHAPE_NEWS, duplicate, *junk])
    )
    news = provider.news("AAPL", limit=5)
    assert [item.title for item in news] == ["Apple beats estimates", "Apple faces EU probe"]
    newest, older = news
    assert newest.publisher == "Reuters" and newest.url == "https://example.com/a"
    assert newest.summary == "Strong iPhone demand."
    assert newest.published_at == datetime(2024, 5, 2, 20, 30, tzinfo=UTC)
    assert older.publisher == "Bloomberg" and older.url == "https://example.com/b"
    assert older.published_at == datetime.fromtimestamp(1714600000, tz=UTC)
    assert provider.news("AAPL", limit=1) == news[:1]


def test_yahoo_news_failure_returns_empty_list() -> None:
    provider, _ = _yahoo(FakeTicker("AAPL", news=RuntimeError("blocked")))  # type: ignore[arg-type]
    assert provider.news("AAPL") == []
    with pytest.raises(InvalidSymbolError):
        provider.news("bad symbol")


def test_parse_news_item_edge_cases() -> None:
    assert parse_news_item(None) is None
    item = parse_news_item(
        {"content": {"title": "T", "pubDate": "garbage", "clickThroughUrl": {"url": "u"}}}
    )
    assert item is not None and item.published_at is None and item.url == "u"
    long = parse_news_item({"title": "x" * 500})
    assert long is not None and len(long.title) == 300


def test_yahoo_uses_yf_ticker_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    created: list[str] = []

    def factory(symbol: str) -> FakeTicker:
        created.append(symbol)
        return FakeTicker(symbol, frame=_yahoo_frame())

    monkeypatch.setattr(yahoo_module.yf, "Ticker", factory)
    provider = YahooFinanceProvider(cache_ttl_seconds=0)
    provider.history("MSFT")
    assert created == ["MSFT"]
    assert isinstance(provider, MarketDataProvider)


# ----------------------------------------------------------------------------- factory


def test_provider_factory(make_settings: Callable[..., Settings]) -> None:
    synthetic = get_market_data_provider(make_settings(data_provider="synthetic", synthetic_seed=9))
    assert isinstance(synthetic, SyntheticProvider) and synthetic.seed == 9
    yahoo = get_market_data_provider(make_settings(data_provider="yahoo", cache_ttl_seconds=12))
    assert isinstance(yahoo, YahooFinanceProvider)
