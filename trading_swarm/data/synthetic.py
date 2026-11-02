"""Deterministic synthetic market data for tests, demos and offline backtests.

Prices follow a seeded geometric Brownian motion (optionally mean-reverting) on a
business-day calendar::

    xₜ = ln Pₜ
    xₜ = xₜ₋₁ + κ·(θ − xₜ₋₁)·Δt + (μₜ − σ²/2)·Δt + σ·√Δt·Zₜ,   Δt = 1/252, Zₜ ~ N(0, 1)

* ``μ`` (``drift``) is the annualised expected return — constant, or piecewise constant
  ("regimes": e.g. a bull run followed by a crash, handy for stop-loss tests);
* ``σ`` (``vol``) is the annualised volatility;
* ``κ`` (``mean_reversion``) > 0 pulls the log-price back towards ``θ``, producing a
  *sideways* market whose stationary spread is ``σ / √(2κ)``.

The path is shifted so that the close on the scenario's ``anchor`` date (default: the first
bar) equals ``start_price`` — this keeps strongly trending scenarios at realistic price
levels around the anchor instead of compounding from an arbitrary origin.

Opens gap from the previous close by a fraction of daily volatility; highs/lows extend
beyond the open/close body; volume scales with the size of the move.

Determinism: every symbol gets independent random streams seeded from ``(seed,
crc32(symbol), stream)``, so a symbol's path never depends on which other symbols were
requested, and — because draws are sequential — extending ``end_date`` never changes
earlier bars (provided the anchor date lies inside the calendar).

Three scenario symbols are built in: ``BULL`` (strong up-trend), ``BEAR`` (strong
down-trend) and ``FLAT`` (mean-reverting range), each with matching canned headlines.
Any other valid ticker gets a reproducible random-but-plausible drift/volatility.
"""

from __future__ import annotations

import logging
import threading
import zlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Final

import numpy as np
import pandas as pd

from trading_swarm.data.base import (
    DateLike,
    MarketDataError,
    NoDataError,
    ProviderUnavailableError,
    normalize_ohlcv,
    slice_range,
    to_utc_date,
    validate_symbol,
)
from trading_swarm.models import NewsItem

logger = logging.getLogger(__name__)

TRADING_DAYS: Final = 252
DEFAULT_ORIGIN: Final = "2021-01-04"
SUPPORTED_INTERVALS: Final = frozenset({"1d", "1wk", "1mo"})
SYNTHETIC_PUBLISHER: Final = "Synthetic Wire"
SCENARIO_ANCHOR: Final = "2026-01-02"

DriftSpec = float | Sequence[tuple[DateLike, float]]

# Stream ids for the independent per-symbol random generators.
_STREAM_CLOSE, _STREAM_GAP, _STREAM_RANGE, _STREAM_VOLUME, _STREAM_NEWS, _STREAM_PARAMS = range(6)

BULLISH_TEMPLATES: Final = (
    "{symbol} beats earnings estimates and raises full-year guidance",
    "{symbol} shares rally to a record high on strong demand",
    "Analysts upgrade {symbol} to buy, citing accelerating revenue growth",
    "{symbol} announces a $5B share buyback and a dividend increase",
    "{symbol} wins a major multi-year contract, boosting its outlook",
    "{symbol} margins expand as costs fall; profit jumps",
)
BEARISH_TEMPLATES: Final = (
    "{symbol} misses earnings estimates and cuts guidance",
    "{symbol} shares plunge after a regulator opens an investigation",
    "Analysts downgrade {symbol} to sell on weakening demand",
    "{symbol} announces layoffs as sales decline for a third quarter",
    "{symbol} faces a lawsuit over accounting concerns",
    "{symbol} warns of losses as debt costs rise",
)
NEUTRAL_TEMPLATES: Final = (
    "{symbol} to present at an industry conference next week",
    "{symbol} schedules its quarterly earnings call",
    "{symbol} names a new chief financial officer",
    "What to watch in {symbol} shares this week",
    "{symbol} trading volume in line with its 30-day average",
    "{symbol} files its annual report with regulators",
)


@dataclass(frozen=True)
class Scenario:
    """Parameters of one synthetic symbol (see the module docstring for the model)."""

    drift: DriftSpec = 0.08
    vol: float = 0.25
    start_price: float = 100.0
    mean_reversion: float = 0.0
    news: tuple[str, ...] = field(default_factory=tuple)
    base_volume: float = 5_000_000.0
    #: Date whose close equals ``start_price`` (``None`` = first bar of the calendar).
    anchor: DateLike | None = None

    def __post_init__(self) -> None:
        if self.vol < 0:
            raise ValueError("vol must be >= 0")
        if self.start_price <= 0:
            raise ValueError("start_price must be > 0")
        if self.mean_reversion < 0:
            raise ValueError("mean_reversion must be >= 0")


DEFAULT_SCENARIOS: Final[Mapping[str, Scenario]] = {
    # Drift/vol ratios are high on purpose: a 20-bar window is bullish (bearish) with
    # ~97 % probability, so trend tests are robust to the pinned end date.
    "BULL": Scenario(
        drift=0.70,
        vol=0.10,
        start_price=100.0,
        anchor=SCENARIO_ANCHOR,
        news=(
            "{symbol} beats earnings estimates and raises full-year guidance",
            "Analysts upgrade {symbol} to buy as revenue growth accelerates",
            "{symbol} shares rally to a record high on surging demand",
            "{symbol} announces a share buyback after a record quarter",
        ),
    ),
    "BEAR": Scenario(
        drift=-0.60,
        vol=0.12,
        start_price=100.0,
        anchor=SCENARIO_ANCHOR,
        news=(
            "{symbol} misses earnings estimates and slashes guidance",
            "Analysts downgrade {symbol} to sell amid weak demand",
            "{symbol} shares plunge as regulators open a fraud investigation",
            "{symbol} announces layoffs after sales decline sharply",
        ),
    ),
    "FLAT": Scenario(
        drift=0.0,
        vol=0.10,
        start_price=100.0,
        mean_reversion=6.0,
        anchor=SCENARIO_ANCHOR,
        news=(
            "{symbol} to present at an industry conference next week",
            "{symbol} schedules its quarterly earnings call",
            "{symbol} files its annual report with regulators",
        ),
    ),
}


def _stable_hash(symbol: str) -> int:
    """Process-independent hash (unlike ``hash()``, which is salted per interpreter)."""
    return zlib.crc32(symbol.encode("utf-8"))


class SyntheticProvider:
    """Offline :class:`~trading_swarm.data.base.MarketDataProvider` with reproducible data.

    Args:
        seed: Global seed; same seed + symbol + calendar ⇒ identical bars.
        drift: Per-symbol annual drift ``μ`` — a float or ``[(from_date, μ), ...]`` regimes.
        vol: Per-symbol annual volatility ``σ``.
        news: Per-symbol headlines (strings may use ``{symbol}``) or :class:`NewsItem`s.
        mean_reversion: Per-symbol ``κ`` (per year) for sideways markets.
        start_price: Per-symbol first close.
        origin: First business day of the generated calendar.
        end_date: Last business day (default: today, UTC). Pin it in tests.
        missing: Symbols that behave like unknown tickers (``NoDataError``).
        failing: Symbols whose requests fail like an outage (``ProviderUnavailableError``).
    """

    name = "synthetic"

    def __init__(
        self,
        seed: int = 42,
        drift: Mapping[str, DriftSpec] | None = None,
        vol: Mapping[str, float] | None = None,
        news: Mapping[str, Sequence[str | NewsItem]] | None = None,
        *,
        mean_reversion: Mapping[str, float] | None = None,
        start_price: Mapping[str, float] | None = None,
        origin: DateLike = DEFAULT_ORIGIN,
        end_date: DateLike | None = None,
        missing: Iterable[str] = (),
        failing: Iterable[str] = (),
    ) -> None:
        self.seed = int(seed)
        self.origin = to_utc_date(origin)
        self.end_date = to_utc_date(end_date if end_date is not None else pd.Timestamp.now("UTC"))
        if self.end_date < self.origin:
            raise ValueError("end_date must not precede origin")
        self._drift = {k.upper(): v for k, v in (drift or {}).items()}
        self._vol = {k.upper(): float(v) for k, v in (vol or {}).items()}
        self._mean_reversion = {k.upper(): float(v) for k, v in (mean_reversion or {}).items()}
        self._start_price = {k.upper(): float(v) for k, v in (start_price or {}).items()}
        self._news: dict[str, list[str | NewsItem]] = {
            k.upper(): list(v) for k, v in (news or {}).items()
        }
        self._missing = {s.upper() for s in missing}
        self._failing = {s.upper() for s in failing}
        self._frames: dict[str, pd.DataFrame] = {}
        self._overrides: dict[str, pd.DataFrame] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ protocol
    def validate_symbol(self, symbol: str) -> str:
        return validate_symbol(symbol)

    def history(
        self,
        symbol: str,
        start: DateLike | None = None,
        end: DateLike | None = None,
        period: str = "1y",
        interval: str = "1d",
    ) -> pd.DataFrame:
        sym = self._check(symbol)
        if interval not in SUPPORTED_INTERVALS:
            raise MarketDataError(
                f"synthetic provider supports intervals {sorted(SUPPORTED_INTERVALS)}, "
                f"not {interval!r}",
                symbol=sym,
            )
        try:
            frame = slice_range(self._frame(sym), start=start, end=end, period=period)
        except ValueError as exc:
            raise MarketDataError(str(exc), symbol=sym) from exc
        if interval != "1d":
            frame = _resample(frame, interval)
        if frame.empty:
            raise NoDataError(f"no synthetic bars for {sym} in the requested range", symbol=sym)
        return frame.copy()

    def latest_price(self, symbol: str) -> float:
        sym = self._check(symbol)
        return float(self._frame(sym)["Close"].iloc[-1])

    def news(self, symbol: str, limit: int = 10) -> list[NewsItem]:
        sym = self._check(symbol, for_news=True)
        if limit <= 0:
            return []
        try:
            frame = self._frame(sym)
        except NoDataError:
            return []
        configured = self._news.get(sym)
        if configured is None:
            scenario = DEFAULT_SCENARIOS.get(sym)
            configured = list(scenario.news) if scenario and scenario.news else None
        titles = configured if configured is not None else self._generated_headlines(sym, frame)
        dates = list(reversed(frame.index[-max(len(titles), 1) :]))
        items: list[NewsItem] = []
        for i, entry in enumerate(titles[:limit]):
            if isinstance(entry, NewsItem):
                items.append(entry)
                continue
            published = dates[min(i, len(dates) - 1)] + pd.Timedelta(hours=13, minutes=30)
            items.append(
                NewsItem(
                    title=entry.format(symbol=sym),
                    publisher=SYNTHETIC_PUBLISHER,
                    url=f"https://example.com/news/{sym.lower()}/{i}",
                    published_at=published.to_pydatetime(),
                    summary=None,
                )
            )
        return items

    # ------------------------------------------------------------------ test helpers
    def set_history(self, symbol: str, df: pd.DataFrame) -> None:
        """Serve ``df`` (normalised) for ``symbol`` instead of generated data."""
        sym = validate_symbol(symbol)
        frame = normalize_ohlcv(df, symbol=sym)
        with self._lock:
            self._overrides[sym] = frame

    def set_news(self, symbol: str, items: Sequence[str | NewsItem]) -> None:
        """Replace the headlines served for ``symbol``."""
        with self._lock:
            self._news[validate_symbol(symbol)] = list(items)

    def scenario(self, symbol: str) -> Scenario:
        """The effective generation parameters for ``symbol``."""
        sym = validate_symbol(symbol)
        base = DEFAULT_SCENARIOS.get(sym) or self._random_scenario(sym)
        updates: dict[str, object] = {}
        if sym in self._drift:
            updates["drift"] = self._drift[sym]
        if sym in self._vol:
            updates["vol"] = self._vol[sym]
        if sym in self._mean_reversion:
            updates["mean_reversion"] = self._mean_reversion[sym]
        if sym in self._start_price:
            updates["start_price"] = self._start_price[sym]
        return replace(base, **updates) if updates else base  # type: ignore[arg-type]

    # ------------------------------------------------------------------ internals
    def _check(self, symbol: str, *, for_news: bool = False) -> str:
        sym = validate_symbol(symbol)
        if sym in self._failing:
            if for_news:
                logger.warning("synthetic news feed unavailable for %s", sym)
                return sym
            raise ProviderUnavailableError(
                f"synthetic outage: data for {sym} is temporarily unavailable", symbol=sym
            )
        if sym in self._missing and not for_news:
            raise NoDataError(f"no price data found for {sym} (unknown symbol?)", symbol=sym)
        return sym

    def _frame(self, sym: str) -> pd.DataFrame:
        if sym in self._missing or sym in self._failing:
            raise NoDataError(f"no price data found for {sym}", symbol=sym)
        with self._lock:
            if sym in self._overrides:
                return self._overrides[sym]
            frame = self._frames.get(sym)
            if frame is None:
                frame = self._generate(sym, self.scenario(sym))
                self._frames[sym] = frame
            return frame

    def _rng(self, sym: str, stream: int) -> np.random.Generator:
        return np.random.default_rng([self.seed, _stable_hash(sym), stream])

    def _random_scenario(self, sym: str) -> Scenario:
        rng = self._rng(sym, _STREAM_PARAMS)
        return Scenario(
            drift=float(rng.uniform(-0.10, 0.30)),
            vol=float(rng.uniform(0.18, 0.45)),
            start_price=float(rng.uniform(20.0, 400.0)),
            base_volume=float(rng.uniform(1e6, 2e7)),
        )

    def _generate(self, sym: str, sc: Scenario) -> pd.DataFrame:
        dates = pd.bdate_range(self.origin, self.end_date, tz="UTC", name="Date")
        n = len(dates)
        if n == 0:
            raise NoDataError(f"empty synthetic calendar for {sym}", symbol=sym)
        dt = 1.0 / TRADING_DAYS
        sigma = sc.vol
        daily_sigma = sigma * np.sqrt(dt)

        mu = _drift_path(sc.drift, dates)
        shocks = self._rng(sym, _STREAM_CLOSE).standard_normal(n)
        increments = (mu - 0.5 * sigma**2) * dt + daily_sigma * shocks
        increments[0] = 0.0
        if sc.mean_reversion > 0:
            # Ornstein-Uhlenbeck in log-price around θ = 0; the anchor shift below moves θ.
            log_close = np.empty(n)
            log_close[0] = 0.0
            keep = 1.0 - sc.mean_reversion * dt
            for t in range(1, n):
                log_close[t] = log_close[t - 1] * keep + increments[t]
        else:
            log_close = np.cumsum(increments)
        anchor_ts = to_utc_date(sc.anchor) if sc.anchor is not None else dates[0]
        anchor_idx = min(int(dates.searchsorted(anchor_ts)), n - 1)
        close = np.exp(log_close - log_close[anchor_idx] + np.log(sc.start_price))

        gaps = self._rng(sym, _STREAM_GAP).standard_normal(n) * daily_sigma * 0.25
        prev_close = np.concatenate(([close[0]], close[:-1]))
        open_ = prev_close * np.exp(gaps)
        ranges = np.abs(self._rng(sym, _STREAM_RANGE).standard_normal((n, 2))) * daily_sigma * 0.5
        high = np.maximum(open_, close) * np.exp(ranges[:, 0])
        low = np.minimum(open_, close) * np.exp(-ranges[:, 1])
        abs_move = np.abs(np.log(close / open_))
        noise = self._rng(sym, _STREAM_VOLUME).standard_normal(n)
        volume = np.round(sc.base_volume * np.exp(0.25 * noise) * (1.0 + 10.0 * abs_move))

        frame = pd.DataFrame(
            {
                "Open": np.round(open_, 4),
                "High": np.round(high, 4),
                "Low": np.round(low, 4),
                "Close": np.round(close, 4),
                "Volume": volume.astype(float),
            },
            index=dates,
        )
        logger.debug("generated %d synthetic bars for %s", n, sym)
        return frame

    def _generated_headlines(self, sym: str, frame: pd.DataFrame) -> list[str]:
        """Canned headlines whose tone follows the trailing 20-bar return."""
        closes = frame["Close"]
        lookback = min(20, len(closes) - 1)
        trailing = float(closes.iloc[-1] / closes.iloc[-1 - lookback] - 1.0) if lookback else 0.0
        if trailing > 0.03:
            tone = BULLISH_TEMPLATES
        elif trailing < -0.03:
            tone = BEARISH_TEMPLATES
        else:
            tone = NEUTRAL_TEMPLATES
        rng = self._rng(sym, _STREAM_NEWS)
        picks = list(rng.permutation(len(tone))[:3])
        neutral_picks = list(rng.permutation(len(NEUTRAL_TEMPLATES))[:2])
        headlines = [tone[i] for i in picks]
        headlines += [
            NEUTRAL_TEMPLATES[i] for i in neutral_picks if NEUTRAL_TEMPLATES[i] not in headlines
        ]
        return headlines


def _drift_path(spec: DriftSpec, dates: pd.DatetimeIndex) -> np.ndarray:
    """Per-bar annual drift for a constant or piecewise-constant regime specification."""
    if isinstance(spec, int | float):
        return np.full(len(dates), float(spec))
    segments = sorted(((to_utc_date(d), float(m)) for d, m in spec), key=lambda s: s[0])
    if not segments:
        raise ValueError("drift regimes must contain at least one (date, drift) pair")
    mu = np.full(len(dates), segments[0][1])
    for starts_on, value in segments:
        mu[dates >= starts_on] = value
    return mu


def _resample(frame: pd.DataFrame, interval: str) -> pd.DataFrame:
    rule = {"1wk": "W-FRI", "1mo": "MS"}[interval]
    agg = frame.resample(rule).agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    )
    return agg.dropna(subset=["Close"])


__all__ = [
    "BEARISH_TEMPLATES",
    "BULLISH_TEMPLATES",
    "DEFAULT_SCENARIOS",
    "NEUTRAL_TEMPLATES",
    "Scenario",
    "SyntheticProvider",
]
