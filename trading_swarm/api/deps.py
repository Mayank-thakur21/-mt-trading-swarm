"""Application wiring: the service container built in the lifespan, and its dependencies.

:func:`build_services` assembles the whole swarm exactly once per process::

    provider     ← DATA_PROVIDER (yahoo | synthetic)
    llm          ← LLM_PROVIDER (Claude by default; lexicon-only when no credentials)
    broker       ← PaperBroker on DATABASE_URL (SQLite)
    agents       ← TrendAnalyzerAgent, SentimentAgent(provider, llm),
                   RiskManagerAgent(cost model of the broker), ExecutionAgent(broker, provider)
    alerts       ← AlertManager(EmailNotifier, broker)
    coordinator  ← SwarmCoordinator(agents, broker, alerts)
    scheduler    ← SwarmScheduler(coordinator)  (only runs with SCHEDULER_ENABLED=true)
    backtester   ← Backtester(provider, llm)    (stateless, shared by all requests)

Endpoints receive the container through the :data:`ServicesDep` dependency, so tests can
inject a synthetic provider, a fake LLM or a pre-seeded broker via
:func:`trading_swarm.api.main.create_app`.
"""

from __future__ import annotations

import contextlib
import logging
import threading
from collections.abc import Iterator
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Annotated, Final

from fastapi import Depends, Request

from trading_swarm.agents import (
    ExecutionAgent,
    RiskManagerAgent,
    SentimentAgent,
    TrendAnalyzerAgent,
)
from trading_swarm.alerts import AlertManager, EmailNotifier
from trading_swarm.api.errors import ApiError
from trading_swarm.backtest import Backtester
from trading_swarm.broker import PaperBroker
from trading_swarm.config import Settings
from trading_swarm.data import MarketDataProvider, get_market_data_provider
from trading_swarm.llm import LLMClient, get_llm_client
from trading_swarm.models import utc_now
from trading_swarm.scheduler import SwarmScheduler
from trading_swarm.swarm import SwarmCoordinator

logger = logging.getLogger(__name__)

#: Backtests are CPU-bound (≈ 13 ms per symbol-bar); more than this many at once would
#: starve the other endpoints of the GIL, so extra requests get ``503 busy``.
MAX_CONCURRENT_BACKTESTS: Final = 2


@dataclass
class ServiceContainer:
    """Long-lived objects shared by every request (built once in the lifespan)."""

    settings: Settings
    provider: MarketDataProvider
    llm: LLMClient
    broker: PaperBroker
    notifier: EmailNotifier
    alerts: AlertManager
    coordinator: SwarmCoordinator
    scheduler: SwarmScheduler
    backtester: Backtester
    owns_broker: bool = True
    started_at: datetime = field(default_factory=utc_now)
    backtest_slots: threading.BoundedSemaphore = field(
        default_factory=lambda: threading.BoundedSemaphore(MAX_CONCURRENT_BACKTESTS)
    )

    def trade_lock(self) -> AbstractContextManager[object]:
        """The coordinator's trading lock, for account mutations made outside it.

        Manual orders and resets take the same lock the coordinator holds around risk
        sizing + execution, so a manual fill can never slip in between an automatic
        trade's risk check and its order (which could overshoot the exposure caps).
        """
        lock = getattr(self.coordinator, "trade_lock", None)
        return lock if lock is not None else contextlib.nullcontext()

    @contextlib.contextmanager
    def backtest_slot(self) -> Iterator[None]:
        """Reserve one of :data:`MAX_CONCURRENT_BACKTESTS` slots or fail with 503."""
        if not self.backtest_slots.acquire(blocking=False):
            raise ApiError(
                503,
                "busy",
                f"{MAX_CONCURRENT_BACKTESTS} backtests are already running; retry shortly",
                headers={"Retry-After": "15"},
                extra={"retry_after": 15},
            )
        try:
            yield
        finally:
            self.backtest_slots.release()

    def close(self) -> None:
        """Stop the scheduler (letting a running scan finish), then release the database.

        The order matters: a scan still running would otherwise use a disposed engine and
        leak sqlite connections.
        """
        try:
            self.scheduler.stop()
        except Exception:  # pragma: no cover - defensive: shutdown must not fail
            logger.exception("error while stopping the scheduler")
        if self.owns_broker:
            try:
                self.broker.close()
            except Exception:  # pragma: no cover - defensive
                logger.exception("error while closing the paper broker")


def build_services(
    settings: Settings,
    *,
    provider: MarketDataProvider | None = None,
    llm: LLMClient | None = None,
    broker: PaperBroker | None = None,
) -> ServiceContainer:
    """Wire the swarm from ``settings``; ``provider`` / ``llm`` / ``broker`` override.

    A broker passed in is *borrowed*: :meth:`ServiceContainer.close` leaves it open.
    """
    market_data = provider if provider is not None else get_market_data_provider(settings)
    llm_client = llm if llm is not None else get_llm_client(settings)
    owns_broker = broker is None
    paper = broker if broker is not None else PaperBroker(settings)
    try:
        notifier = EmailNotifier(settings)
        alerts = AlertManager(notifier, paper, settings)
        coordinator = SwarmCoordinator(
            settings,
            market_data,
            paper,
            llm=llm_client,
            alerts=alerts,
            trend=TrendAnalyzerAgent(settings),
            sentiment=SentimentAgent(market_data, llm_client, settings),
            risk=RiskManagerAgent(settings, cost_model=paper.cost_model),
            execution=ExecutionAgent(paper, settings, market_data),
        )
        scheduler = SwarmScheduler(coordinator, settings)
        backtester = Backtester(market_data, settings, llm=llm_client)
    except Exception:
        if owns_broker:
            paper.close()
        raise
    return ServiceContainer(
        settings=settings,
        provider=market_data,
        llm=llm_client,
        broker=paper,
        notifier=notifier,
        alerts=alerts,
        coordinator=coordinator,
        scheduler=scheduler,
        backtester=backtester,
        owns_broker=owns_broker,
    )


def get_services(request: Request) -> ServiceContainer:
    """The container built by the lifespan (``503`` while starting up / shutting down)."""
    services: ServiceContainer | None = getattr(request.app.state, "services", None)
    if services is None:
        raise ApiError(
            503,
            "service_unavailable",
            "the service is starting up or shutting down; retry shortly",
            headers={"Retry-After": "5"},
        )
    return services


def get_settings_dep(request: Request) -> Settings:
    """The settings the application was created with."""
    return request.app.state.settings  # type: ignore[no-any-return]


#: Inject the :class:`ServiceContainer` into an endpoint.
ServicesDep = Annotated[ServiceContainer, Depends(get_services)]
#: Inject the application's :class:`~trading_swarm.config.Settings`.
SettingsDep = Annotated[Settings, Depends(get_settings_dep)]

__all__ = [
    "MAX_CONCURRENT_BACKTESTS",
    "ServiceContainer",
    "ServicesDep",
    "SettingsDep",
    "build_services",
    "get_services",
    "get_settings_dep",
]
