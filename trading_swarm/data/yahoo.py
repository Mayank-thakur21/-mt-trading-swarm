"""Yahoo Finance market data via :mod:`yfinance`, with a TTL cache and retries.

* Prices are **split- and dividend-adjusted** (``auto_adjust=True``) so indicators and
  returns are not distorted by corporate actions.
* Failures are classified: an empty answer for a well-formed ticker is a
  :class:`NoDataError` (unknown/delisted symbol — not retried), while exceptions such as
  timeouts or rate limiting are retried with exponential back-off and finally surface as
  :class:`ProviderUnavailableError`.
* ``Ticker.news`` changed shape around yfinance 0.2.5x: new items nest everything under
  ``"content"`` (``title``, ``summary``, ``pubDate``, ``provider.displayName``,
  ``canonicalUrl.url``); older ones are flat (``title``, ``publisher``, ``link``,
  ``providerPublishTime``). :func:`parse_news_item` accepts both.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import Any, Final

import pandas as pd
import yfinance as yf

from trading_swarm.config import Settings
from trading_swarm.data.base import (
    VALID_PERIODS,
    DateLike,
    MarketDataError,
    NoDataError,
    ProviderUnavailableError,
    apply_period,
    normalize_ohlcv,
    to_utc_date,
    validate_symbol,
)
from trading_swarm.data.cache import TTLCache
from trading_swarm.models import NewsItem

logger = logging.getLogger(__name__)

QUOTE_TTL_CAP_SECONDS: Final = 60.0
_NO_DATA_EXCEPTIONS: tuple[type[BaseException], ...] = tuple(
    exc
    for exc in (
        getattr(getattr(yf, "exceptions", None), name, None)
        for name in ("YFTickerMissingError", "YFPricesMissingError", "YFTzMissingError")
    )
    if isinstance(exc, type)
)


def _period_window_start(end: pd.Timestamp, period: str) -> pd.Timestamp | None:
    """A start date safely before the first bar of ``period`` ending at ``end``."""
    if period not in VALID_PERIODS:
        raise MarketDataError(f"unsupported period {period!r}")
    if period == "max":
        return None
    if period in {"1d", "5d"}:
        return end - pd.Timedelta(days=14)
    if period == "ytd":
        return pd.Timestamp(year=end.year, month=1, day=1, tz="UTC") - pd.Timedelta(days=7)
    if period.endswith("mo"):
        return end - pd.DateOffset(months=int(period[:-2])) - pd.Timedelta(days=7)
    return end - pd.DateOffset(years=int(period[:-1])) - pd.Timedelta(days=7)


def _parse_timestamp(value: Any) -> datetime | None:
    """Epoch seconds (old news shape) or ISO-8601 strings (new shape) → aware UTC."""
    if value is None or value == "":
        return None
    try:
        if isinstance(value, int | float) and not isinstance(value, bool):
            if not math.isfinite(value):
                return None
            return datetime.fromtimestamp(float(value), tz=UTC)
        ts = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError, OSError):
        return None
    if pd.isna(ts):
        return None
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return ts.to_pydatetime()


def _clean_text(value: Any, limit: int) -> str | None:
    if not isinstance(value, str):
        return None
    text = " ".join(value.split())
    if not text:
        return None
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def parse_news_item(raw: Any) -> NewsItem | None:
    """Convert one ``Ticker.news`` entry (either shape) into a :class:`NewsItem`."""
    if not isinstance(raw, Mapping):
        return None
    content = raw.get("content")
    if isinstance(content, Mapping):
        provider = content.get("provider")
        canonical = content.get("canonicalUrl")
        click = content.get("clickThroughUrl")
        url = None
        for link in (canonical, click):
            if isinstance(link, Mapping) and isinstance(link.get("url"), str):
                url = link["url"]
                break
        title = content.get("title")
        publisher = provider.get("displayName") if isinstance(provider, Mapping) else None
        published = content.get("pubDate") or content.get("displayTime")
        summary = content.get("summary") or content.get("description")
    else:
        title = raw.get("title")
        publisher = raw.get("publisher")
        url = raw.get("link") if isinstance(raw.get("link"), str) else None
        published = raw.get("providerPublishTime")
        summary = raw.get("summary")
    clean_title = _clean_text(title, 300)
    if clean_title is None:
        return None
    return NewsItem(
        title=clean_title,
        publisher=_clean_text(publisher, 120),
        url=url,
        published_at=_parse_timestamp(published),
        summary=_clean_text(summary, 1000),
    )


class YahooFinanceProvider:
    """:class:`~trading_swarm.data.base.MarketDataProvider` backed by Yahoo Finance.

    Args:
        cache_ttl_seconds: How long downloads are reused (quotes are capped at 60 s).
        max_retries: Extra attempts after the first failure (exponential back-off).
        timeout: Per-request timeout in seconds passed to yfinance.
        backoff_seconds: First back-off delay; doubles on each retry.
        ticker_factory: Builds ``yf.Ticker`` objects (injectable for tests).
        sleep: Sleep function (injectable so tests do not wait).
    """

    name = "yahoo"

    def __init__(
        self,
        cache_ttl_seconds: float = 300,
        max_retries: int = 2,
        timeout: float = 10.0,
        backoff_seconds: float = 0.5,
        ticker_factory: Callable[[str], Any] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.max_retries = max(0, int(max_retries))
        self.timeout = float(timeout)
        self.backoff_seconds = max(0.0, float(backoff_seconds))
        self._ticker_factory = ticker_factory
        self._sleep = sleep
        self._history_cache: TTLCache[pd.DataFrame] = TTLCache(cache_ttl_seconds)
        self._quote_cache: TTLCache[float] = TTLCache(min(cache_ttl_seconds, QUOTE_TTL_CAP_SECONDS))
        self._news_cache: TTLCache[list[NewsItem]] = TTLCache(cache_ttl_seconds)

    @classmethod
    def from_settings(cls, settings: Settings) -> YahooFinanceProvider:
        return cls(cache_ttl_seconds=settings.cache_ttl_seconds)

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
        sym = validate_symbol(symbol)
        if start is None and period not in VALID_PERIODS:
            raise MarketDataError(
                f"unsupported period {period!r}; choose one of {sorted(VALID_PERIODS)}",
                symbol=sym,
            )
        start_ts = to_utc_date(start) if start is not None else None
        # Our ``end`` is inclusive; yfinance's is exclusive.
        end_ts = to_utc_date(end) + pd.Timedelta(days=1) if end is not None else None
        key = ("history", sym, start_ts, end_ts, None if start_ts else period, interval)

        def download() -> pd.DataFrame:
            kwargs: dict[str, Any] = {
                "interval": interval,
                "auto_adjust": True,
                "actions": False,
                "timeout": self.timeout,
                "raise_errors": True,
            }
            window_start = start_ts
            if window_start is None and end_ts is not None:
                # A trailing ``period`` that ends at ``end``: download a generous window
                # and trim it with the same period semantics as the other providers.
                window_start = _period_window_start(end_ts, period)
            if window_start is not None:
                kwargs["start"] = window_start.strftime("%Y-%m-%d")
                if end_ts is not None:
                    kwargs["end"] = end_ts.strftime("%Y-%m-%d")
            else:
                kwargs["period"] = period
            raw = self._call(sym, "history", lambda: self._ticker(sym).history(**kwargs))
            frame = normalize_ohlcv(raw, symbol=sym, interval=interval)
            if start_ts is None and end_ts is not None:
                frame = apply_period(frame[frame.index < end_ts], period)
            if frame.empty:
                raise NoDataError(f"no price data for {sym} in the requested range", symbol=sym)
            return frame

        return self._history_cache.get_or_set(key, download).copy(deep=False)

    def latest_price(self, symbol: str) -> float:
        sym = validate_symbol(symbol)

        def fetch() -> float:
            price = self._fast_info_price(sym)
            if price is not None:
                return price
            frame = self.history(sym, period="5d")
            return float(frame["Close"].iloc[-1])

        return self._quote_cache.get_or_set(("quote", sym), fetch)

    def news(self, symbol: str, limit: int = 10) -> list[NewsItem]:
        sym = validate_symbol(symbol)
        if limit <= 0:
            return []

        def fetch() -> list[NewsItem]:
            ticker = self._ticker(sym)
            getter = getattr(ticker, "get_news", None)
            raw = getter(count=max(limit, 10)) if callable(getter) else ticker.news
            items: list[NewsItem] = []
            seen: set[str] = set()
            for entry in raw or []:
                item = parse_news_item(entry)
                if item is None or item.title.lower() in seen:
                    continue
                seen.add(item.title.lower())
                items.append(item)
            epoch = datetime.min.replace(tzinfo=UTC)
            items.sort(key=lambda it: it.published_at or epoch, reverse=True)
            return items

        try:
            items = self._news_cache.get_or_set(("news", sym), fetch)
        except Exception as exc:
            logger.warning("news unavailable for %s: %s", sym, exc)
            return []
        return list(items[:limit])

    # ------------------------------------------------------------------ internals
    def _ticker(self, symbol: str) -> Any:
        factory = self._ticker_factory or yf.Ticker
        return factory(symbol)

    def _fast_info_price(self, sym: str) -> float | None:
        try:
            info = self._ticker(sym).fast_info
            price: Any = None
            for key in ("lastPrice", "last_price", "regularMarketPrice"):
                try:
                    price = info[key]
                except (KeyError, TypeError, AttributeError):
                    price = getattr(info, key, None)
                if price is not None:
                    break
            value = float(price) if price is not None else None
        except Exception as exc:  # fast_info does lazy network I/O and raises many types
            logger.debug("fast_info failed for %s: %s", sym, exc)
            return None
        if value is None or not math.isfinite(value) or value <= 0:
            return None
        return value

    def _call[T](self, sym: str, what: str, fn: Callable[[], T]) -> T:
        """Run ``fn`` with retries; classify failures into MarketDataError subclasses."""
        attempts = self.max_retries + 1
        last_exc: BaseException | None = None
        for attempt in range(1, attempts + 1):
            try:
                return fn()
            except MarketDataError:
                raise
            except _NO_DATA_EXCEPTIONS as exc:
                raise NoDataError(f"no data for {sym}: {exc}", symbol=sym) from exc
            except Exception as exc:
                last_exc = exc
                if attempt < attempts:
                    delay = self.backoff_seconds * (2 ** (attempt - 1))
                    logger.warning(
                        "yahoo %s for %s failed (attempt %d/%d): %s; retrying in %.1fs",
                        what,
                        sym,
                        attempt,
                        attempts,
                        exc,
                        delay,
                    )
                    self._sleep(delay)
        logger.error("yahoo %s for %s failed after %d attempts: %s", what, sym, attempts, last_exc)
        raise ProviderUnavailableError(
            f"Yahoo Finance {what} request for {sym} failed after {attempts} attempts: {last_exc}",
            symbol=sym,
        ) from last_exc


__all__ = ["YahooFinanceProvider", "parse_news_item"]
