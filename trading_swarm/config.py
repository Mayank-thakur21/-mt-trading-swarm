"""Application settings, loaded from environment variables and an optional ``.env`` file.

Every knob of the swarm lives here so that the API, the scheduler, the backtester and the
dashboard all agree on the same numbers. Values are validated at start-up: a mistyped risk
limit (say ``RISK_PER_TRADE=1`` meaning "1 %") fails fast instead of silently risking the
whole account on one trade.

Units convention
----------------
* Risk limits (``risk_per_trade``, ``max_position_pct``, ``max_gross_exposure``,
  ``max_drawdown_halt``, ``max_volatility``) are **fractions**: ``0.01`` means 1 %.
* ``slippage_bps`` is in basis points (1 bp = 0.01 %).
* ``alert_price_move_pct`` is a **percentage** (``5.0`` means a 5 % move), matching how
  traders phrase price-move alerts.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

DataProviderName = Literal["yahoo", "synthetic"]
LLMProviderName = Literal["anthropic", "openai", "none"]
EffortLevel = Literal["low", "medium", "high", "xhigh", "max"]


def _split_csv(raw: str, *, upper: bool = False) -> list[str]:
    """Split a comma-separated env value into a de-duplicated, order-preserving list."""
    seen: dict[str, None] = {}
    for part in raw.split(","):
        item = part.strip()
        if upper:
            item = item.upper()
        if item:
            seen.setdefault(item, None)
    return list(seen)


class Settings(BaseSettings):
    """Typed configuration for the whole application.

    Field names map to upper-case environment variables (``risk_per_trade`` ←
    ``RISK_PER_TRADE``); the only exception is :attr:`env`, read from ``APP_ENV``.
    Unknown variables are ignored so a shared ``.env`` can hold dashboard-only keys too.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        validate_by_name=True,
        validate_by_alias=True,
    )

    # ----------------------------------------------------------------- application
    env: str = Field(default="development", validation_alias="APP_ENV")
    log_level: str = "INFO"
    log_json: bool = False

    # ----------------------------------------------------------------- providers
    data_provider: DataProviderName = "yahoo"
    synthetic_seed: int = 42
    cache_ttl_seconds: int = Field(default=300, ge=0)

    llm_provider: LLMProviderName = "anthropic"
    anthropic_api_key: str = Field(default="", repr=False)
    anthropic_use_sdk_default_auth: bool = False
    anthropic_model: str = "claude-opus-5-5"
    anthropic_effort: EffortLevel = "low"
    openai_api_key: str = Field(default="", repr=False)
    openai_model: str = "gpt-5-mini"
    llm_timeout_seconds: float = Field(default=30.0, gt=0)
    llm_max_retries: int = Field(default=2, ge=0, le=10)

    # ----------------------------------------------------------------- storage
    database_url: str = "sqlite:///./data/trading_swarm.db"

    # ----------------------------------------------------------------- account & costs
    initial_capital: float = Field(default=100_000.0, gt=0)
    commission_rate: float = Field(default=0.0005, ge=0, lt=0.1)
    min_commission: float = Field(default=1.0, ge=0)
    slippage_bps: float = Field(default=5.0, ge=0, lt=1_000)

    # ----------------------------------------------------------------- risk
    risk_per_trade: float = Field(default=0.01, gt=0, le=0.1)
    max_position_pct: float = Field(default=0.20, gt=0, le=1.0)
    max_gross_exposure: float = Field(default=0.95, gt=0, le=1.0)
    atr_stop_multiple: float = Field(default=2.0, gt=0, le=20)
    take_profit_r: float = Field(default=2.0, gt=0, le=50)
    max_drawdown_halt: float = Field(default=0.20, gt=0, le=1.0)
    max_volatility: float = Field(default=1.20, gt=0)
    risk_free_rate: float = Field(default=0.0, ge=-0.1, le=1.0)

    # ----------------------------------------------------------------- consensus
    buy_threshold: float = Field(default=0.25, ge=-1, le=1)
    sell_threshold: float = Field(default=-0.25, ge=-1, le=1)
    weight_trend: float = Field(default=0.65, ge=0)
    weight_sentiment: float = Field(default=0.35, ge=0)
    auto_trade: bool = False
    watchlist: str = "AAPL,MSFT,GOOGL,AMZN,NVDA"

    # ----------------------------------------------------------------- scheduler
    scheduler_enabled: bool = False
    scan_interval_minutes: int = Field(default=60, ge=1)
    market_hours_only: bool = False

    # ----------------------------------------------------------------- API
    api_keys: str = Field(default="", repr=False)
    rate_limit_per_minute: int = Field(default=60, ge=1)
    #: Browser origins allowed to call the API (comma list, ``*`` = any). Empty (the
    #: default) = none: the Streamlit dashboard calls the API from its server, so it
    #: needs no CORS, and cross-site pages cannot drive the paper account.
    cors_origins: str = ""
    dashboard_url: str = "http://localhost:8501"

    # ----------------------------------------------------------------- e-mail alerts
    smtp_host: str = ""
    smtp_port: int = Field(default=587, ge=1, le=65535)
    smtp_username: str = ""
    smtp_password: str = Field(default="", repr=False)
    smtp_use_tls: bool = True
    smtp_use_ssl: bool = False
    alert_email_from: str = ""
    alert_email_to: str = ""
    alert_min_confidence: float = Field(default=0.6, ge=0, le=1)
    alert_price_move_pct: float = Field(default=5.0, gt=0)
    alert_cooldown_minutes: int = Field(default=60, ge=0)

    # ----------------------------------------------------------------- validators
    @field_validator("log_level")
    @classmethod
    def _normalise_log_level(cls, value: str) -> str:
        level = value.strip().upper()
        if level not in {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}:
            raise ValueError(f"unknown log level {value!r}")
        return level

    @field_validator("anthropic_api_key", "openai_api_key", "smtp_host", "alert_email_from")
    @classmethod
    def _strip(cls, value: str) -> str:
        return value.strip()

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        if self.sell_threshold >= self.buy_threshold:
            raise ValueError(
                f"SELL_THRESHOLD ({self.sell_threshold}) must be below "
                f"BUY_THRESHOLD ({self.buy_threshold})"
            )
        if self.weight_trend + self.weight_sentiment <= 0:
            raise ValueError("at least one of WEIGHT_TREND / WEIGHT_SENTIMENT must be > 0")
        if self.smtp_use_ssl and self.smtp_use_tls:
            # Implicit TLS (SMTPS, port 465) and STARTTLS are mutually exclusive; SSL wins.
            self.smtp_use_tls = False
        return self

    # ----------------------------------------------------------------- derived values
    @property
    def watchlist_symbols(self) -> list[str]:
        """Upper-cased, de-duplicated watchlist from the comma-separated ``WATCHLIST``."""
        return _split_csv(self.watchlist, upper=True)

    @property
    def api_key_set(self) -> frozenset[str]:
        """The accepted API keys (``API_KEYS`` comma list)."""
        return frozenset(_split_csv(self.api_keys))

    @property
    def auth_enabled(self) -> bool:
        """``True`` when at least one API key is configured; otherwise the API is open."""
        return bool(self.api_key_set)

    @property
    def alert_recipients(self) -> list[str]:
        """Recipient addresses from the comma-separated ``ALERT_EMAIL_TO``."""
        return _split_csv(self.alert_email_to)

    @property
    def cors_origin_list(self) -> list[str]:
        """Allowed browser origins (``["*"]`` allows any; empty allows none).

        Origins are compared without a trailing slash (``http://host:8501``).
        """
        return [origin.rstrip("/") for origin in _split_csv(self.cors_origins)]

    @property
    def email_enabled(self) -> bool:
        """E-mail alerts need an SMTP host, a sender and at least one recipient."""
        return bool(self.smtp_host and self.alert_email_from and self.alert_recipients)

    @property
    def anthropic_credentials_available(self) -> bool:
        """An explicit key, or permission to let the SDK resolve its default credentials."""
        return bool(self.anthropic_api_key) or self.anthropic_use_sdk_default_auth

    @property
    def effective_llm_provider(self) -> LLMProviderName:
        """The provider that will actually be used once missing credentials are considered.

        Mirrors :func:`trading_swarm.llm.get_llm_client` (which also logs a warning): a
        provider without credentials degrades to ``"none"`` (lexicon-only sentiment).
        """
        if self.llm_provider == "anthropic" and not self.anthropic_credentials_available:
            return "none"
        if self.llm_provider == "openai" and not self.openai_api_key:
            return "none"
        return self.llm_provider

    @property
    def llm_model(self) -> str:
        """Model id of the effective LLM provider (empty string when disabled)."""
        provider = self.effective_llm_provider
        if provider == "anthropic":
            return self.anthropic_model
        if provider == "openai":
            return self.openai_model
        return ""

    @property
    def is_production(self) -> bool:
        return self.env.lower() in {"prod", "production"}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide :class:`Settings` (cached; call ``get_settings.cache_clear()``
    in tests after changing the environment)."""
    return Settings()


__all__ = [
    "DataProviderName",
    "EffortLevel",
    "LLMProviderName",
    "Settings",
    "get_settings",
]
