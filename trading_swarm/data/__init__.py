"""Market-data providers (Yahoo Finance for live data, a seeded synthetic generator offline)."""

from __future__ import annotations

from trading_swarm.config import Settings
from trading_swarm.data.base import (
    DateLike,
    InvalidSymbolError,
    MarketDataError,
    MarketDataProvider,
    NoDataError,
    ProviderUnavailableError,
    daily_change_pct,
    validate_symbol,
)
from trading_swarm.data.synthetic import DEFAULT_SCENARIOS, Scenario, SyntheticProvider
from trading_swarm.data.yahoo import YahooFinanceProvider


def get_market_data_provider(settings: Settings) -> MarketDataProvider:
    """Build the provider selected by ``DATA_PROVIDER`` (``yahoo`` or ``synthetic``)."""
    if settings.data_provider == "synthetic":
        return SyntheticProvider(seed=settings.synthetic_seed)
    return YahooFinanceProvider.from_settings(settings)


__all__ = [
    "DEFAULT_SCENARIOS",
    "DateLike",
    "InvalidSymbolError",
    "MarketDataError",
    "MarketDataProvider",
    "NoDataError",
    "ProviderUnavailableError",
    "Scenario",
    "SyntheticProvider",
    "YahooFinanceProvider",
    "daily_change_pct",
    "get_market_data_provider",
    "validate_symbol",
]
