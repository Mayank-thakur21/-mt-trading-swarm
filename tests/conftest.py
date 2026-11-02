"""Shared pytest fixtures. Nothing here touches the network.

Fixtures
--------
``settings`` / ``make_settings``
    Isolated :class:`Settings`: temp SQLite file, ``data_provider="synthetic"``,
    ``llm_provider="none"``, watchlist ``BULL,BEAR,FLAT``; ``.env`` files and the real
    environment are ignored.
``synthetic_provider`` (alias ``provider``)
    :class:`SyntheticProvider` pinned to :data:`TEST_END_DATE`. ``BULL`` / ``BEAR`` /
    ``FLAT`` are the built-in trending-up / trending-down / range-bound scenarios;
    ``NOPE`` behaves like an unknown ticker (``NoDataError``) and ``DOWN`` like an outage
    (``ProviderUnavailableError``).
``broker``
    :class:`PaperBroker` on the temp database (closed after the test).
``fake_llm`` / ``fake_llm_factory`` / ``make_judgement``
    A :class:`FakeLLM` returning a configurable :class:`SentimentJudgement` (or ``None``).
``fake_anthropic_client`` / ``make_anthropic_response`` / ``make_anthropic_error``
    Test doubles for the ``anthropic`` SDK (``client.beta.messages.create``).
``fake_openai_client`` / ``make_openai_completion`` / ``make_openai_error``
    Test doubles for the ``openai`` SDK (``client.chat.completions.create``).
``fake_smtp``
    Monkeypatches ``smtplib.SMTP`` / ``smtplib.SMTP_SSL`` with recording fakes.
``make_ohlcv``
    Builds a canonical OHLCV frame from a list of closes.

At :data:`TEST_END_DATE` the scenarios read cleanly: BULL is above its SMA20/50/200 with a
golden cross, positive and rising MACD histogram, positive 20-day momentum and ADX ≈ 63
(RSI ≈ 85, overbought); BEAR is the mirror image (RSI ≈ 19, ADX ≈ 45); FLAT has ≈ 0 %
20-day momentum and ADX ≈ 21.
"""

from __future__ import annotations

import json
import smtplib
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import anthropic
import httpx2
import numpy as np
import openai
import pandas as pd
import pytest

from trading_swarm.broker import PaperBroker
from trading_swarm.config import Settings, get_settings
from trading_swarm.data import SyntheticProvider
from trading_swarm.models import HeadlineSentiment, NewsItem, SentimentJudgement

TEST_END_DATE = "2026-09-14"
TEST_SEED = 42

# Variables read by the SDKs themselves; cleared so no test can pick up real credentials.
_SDK_ENV_VARS = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_PROFILE",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_ORG_ID",
)


def _settings_env_names() -> set[str]:
    names = set()
    for name, info in Settings.model_fields.items():
        alias = info.validation_alias
        names.add(alias.upper() if isinstance(alias, str) else name.upper())
    return names


@pytest.fixture(autouse=True)
def _isolated_environment(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Every test starts from defaults: no env overrides, no ``.env``, fresh cache."""
    for var in _settings_env_names() | set(_SDK_ENV_VARS):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setitem(Settings.model_config, "env_file", None)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


# =============================================================================== settings


@pytest.fixture
def make_settings(tmp_path: Path) -> Callable[..., Settings]:
    """Factory: ``make_settings(**overrides)`` → isolated :class:`Settings`."""

    def factory(**overrides: Any) -> Settings:
        values: dict[str, Any] = {
            "env": "test",
            "database_url": f"sqlite:///{tmp_path / 'trading_swarm.db'}",
            "data_provider": "synthetic",
            "llm_provider": "none",
            "watchlist": "BULL,BEAR,FLAT",
        }
        values.update(overrides)
        return Settings(_env_file=None, **values)

    return factory


@pytest.fixture
def settings(make_settings: Callable[..., Settings]) -> Settings:
    return make_settings()


# =============================================================================== market data


@pytest.fixture
def synthetic_provider() -> SyntheticProvider:
    return SyntheticProvider(
        seed=TEST_SEED, end_date=TEST_END_DATE, missing={"NOPE"}, failing={"DOWN"}
    )


@pytest.fixture
def provider(synthetic_provider: SyntheticProvider) -> SyntheticProvider:
    """Alias of :func:`synthetic_provider`."""
    return synthetic_provider


@pytest.fixture
def make_ohlcv() -> Callable[..., pd.DataFrame]:
    """Factory: ``make_ohlcv(closes, start="2024-01-02", spread=0.01, volume=1e6)``.

    Opens equal the previous close; highs/lows sit ``spread`` above/below the body.
    """

    def factory(
        closes: Sequence[float],
        start: str = "2024-01-02",
        spread: float = 0.01,
        volume: float = 1_000_000.0,
    ) -> pd.DataFrame:
        close = np.asarray(closes, dtype=float)
        index = pd.bdate_range(start, periods=len(close), tz="UTC", name="Date")
        open_ = np.concatenate(([close[0]], close[:-1]))
        high = np.maximum(open_, close) * (1 + spread)
        low = np.minimum(open_, close) * (1 - spread)
        return pd.DataFrame(
            {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
            index=index,
        )

    return factory


# =============================================================================== broker


@pytest.fixture
def broker(settings: Settings) -> Iterator[PaperBroker]:
    paper = PaperBroker(settings)
    try:
        yield paper
    finally:
        paper.close()


# =============================================================================== fake LLM


def _label(score: float) -> str:
    if score > 0.1:
        return "bullish"
    if score < -0.1:
        return "bearish"
    return "neutral"


def build_judgement(
    score: float = 0.6,
    confidence: float = 0.8,
    n: int = 3,
    relevance: float = 1.0,
    summary: str = "Headlines point to strong demand and raised guidance.",
) -> SentimentJudgement:
    items = [
        HeadlineSentiment(index=i, sentiment=_label(score), score=score, relevance=relevance)
        for i in range(n)
    ]
    return SentimentJudgement(
        items=items, overall_score=score, confidence=confidence, summary=summary
    )


class FakeLLM:
    """An :class:`~trading_swarm.llm.base.LLMClient` that returns ``judgement`` verbatim."""

    def __init__(
        self,
        judgement: SentimentJudgement | None = None,
        *,
        name: str = "fake",
        model: str = "fake-model",
    ) -> None:
        self.name = name
        self.model = model
        self.judgement = judgement
        self.calls: list[tuple[str, list[NewsItem]]] = []

    def judge_sentiment(
        self, symbol: str, headlines: Sequence[NewsItem]
    ) -> SentimentJudgement | None:
        self.calls.append((symbol, list(headlines)))
        return self.judgement


@pytest.fixture
def make_judgement() -> Callable[..., SentimentJudgement]:
    """Factory: ``make_judgement(score=0.6, confidence=0.8, n=3, relevance=1.0)``."""
    return build_judgement


@pytest.fixture
def fake_llm_factory() -> type[FakeLLM]:
    """The :class:`FakeLLM` class: ``fake_llm_factory(judgement, name=..., model=...)``."""
    return FakeLLM


@pytest.fixture
def fake_llm() -> FakeLLM:
    """A bullish fake LLM (score 0.6, confidence 0.8); set ``.judgement = None`` to fail."""
    return FakeLLM(build_judgement())


# =============================================================================== fake anthropic


class _FakeMessages:
    def __init__(self, response: Any = None, error: BaseException | None = None) -> None:
        self.response = response
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


class FakeAnthropicClient:
    """Mimics ``anthropic.Anthropic`` far enough for ``client.beta.messages.create``."""

    def __init__(self, response: Any = None, *, error: BaseException | None = None) -> None:
        self.messages = _FakeMessages(response, error)
        self.beta = SimpleNamespace(messages=self.messages)

    @property
    def calls(self) -> list[dict[str, Any]]:
        return self.messages.calls


def build_anthropic_response(
    payload: dict[str, Any] | str | None = None,
    *,
    stop_reason: str = "end_turn",
    stop_details: Any = None,
    request_id: str = "req_test_123",
    include_thinking: bool = True,
) -> SimpleNamespace:
    """A ``BetaMessage``-like object; ``payload`` dicts are JSON-encoded into a text block."""
    if payload is None:
        payload = build_judgement().model_dump(mode="json")
    content: list[SimpleNamespace] = []
    if include_thinking:
        content.append(SimpleNamespace(type="thinking", thinking="", signature="sig"))
    text = payload if isinstance(payload, str) else json.dumps(payload)
    content.append(SimpleNamespace(type="text", text=text))
    return SimpleNamespace(
        id="msg_test",
        model="claude-opus-5-5",
        stop_reason=stop_reason,
        stop_details=stop_details,
        content=content,
        usage=SimpleNamespace(input_tokens=321, output_tokens=123),
        _request_id=request_id,
    )


def build_status_error(
    error_cls: type[Exception], status: int, message: str = "boom", request_id: str = "req_err_1"
) -> Exception:
    """Instantiate an SDK ``APIStatusError`` subclass with a fake HTTP response."""
    request = httpx2.Request("POST", "https://api.example.invalid/v1/test")
    response = httpx2.Response(
        status, request=request, headers={"request-id": request_id, "x-request-id": request_id}
    )
    return error_cls(message, response=response, body=None)


@pytest.fixture
def fake_anthropic_client() -> type[FakeAnthropicClient]:
    """The fake client class: ``fake_anthropic_client(response, error=...)``."""
    return FakeAnthropicClient


@pytest.fixture
def make_anthropic_response() -> Callable[..., SimpleNamespace]:
    return build_anthropic_response


@pytest.fixture
def make_anthropic_error() -> Callable[..., Exception]:
    """Factory: ``make_anthropic_error(anthropic.RateLimitError, 429)`` (or a connection
    error with ``make_anthropic_error(anthropic.APIConnectionError)``)."""

    def factory(error_cls: type[Exception], status: int | None = None, **kwargs: Any) -> Exception:
        if issubclass(error_cls, anthropic.APIConnectionError):
            request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
            return error_cls(request=request)
        return build_status_error(error_cls, status or 500, **kwargs)

    return factory


# =============================================================================== fake openai


class _FakeCompletions:
    def __init__(self, response: Any = None, error: BaseException | None = None) -> None:
        self.response = response
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


class FakeOpenAIClient:
    """Mimics ``openai.OpenAI`` far enough for ``client.chat.completions.create``."""

    def __init__(self, response: Any = None, *, error: BaseException | None = None) -> None:
        self.completions = _FakeCompletions(response, error)
        self.chat = SimpleNamespace(completions=self.completions)

    @property
    def calls(self) -> list[dict[str, Any]]:
        return self.completions.calls


def build_openai_completion(
    payload: dict[str, Any] | str | None = None,
    *,
    finish_reason: str = "stop",
    refusal: str | None = None,
    request_id: str = "req_oai_1",
) -> SimpleNamespace:
    if payload is None:
        payload = build_judgement().model_dump(mode="json")
    content = None if refusal else (payload if isinstance(payload, str) else json.dumps(payload))
    message = SimpleNamespace(role="assistant", content=content, refusal=refusal)
    return SimpleNamespace(
        id="chatcmpl-test",
        choices=[SimpleNamespace(index=0, finish_reason=finish_reason, message=message)],
        _request_id=request_id,
    )


@pytest.fixture
def fake_openai_client() -> type[FakeOpenAIClient]:
    return FakeOpenAIClient


@pytest.fixture
def make_openai_completion() -> Callable[..., SimpleNamespace]:
    return build_openai_completion


@pytest.fixture
def make_openai_error() -> Callable[..., Exception]:
    def factory(error_cls: type[Exception], status: int | None = None, **kwargs: Any) -> Exception:
        if issubclass(error_cls, openai.APIConnectionError):
            request = httpx2.Request("POST", "https://api.openai.com/v1/chat/completions")
            return error_cls(request=request)
        return build_status_error(error_cls, status or 500, **kwargs)

    return factory


# =============================================================================== fake SMTP


@dataclass
class FakeSMTPController:
    """Records every fake SMTP connection; set ``fail_with`` to make ``send_message`` raise."""

    instances: list[FakeSMTP] = field(default_factory=list)
    fail_with: BaseException | None = None

    @property
    def sent(self) -> list[Any]:
        return [msg for inst in self.instances for msg in inst.messages]


class FakeSMTP:
    controller: FakeSMTPController
    ssl = False

    def __init__(self, host: str = "", port: int = 0, *args: Any, **kwargs: Any) -> None:
        self.host = host
        self.port = port
        self.timeout = kwargs.get("timeout")
        self.started_tls = False
        self.logged_in_as: str | None = None
        self.messages: list[Any] = []
        self.closed = False
        self.controller.instances.append(self)

    def __enter__(self) -> FakeSMTP:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.quit()

    def ehlo(self, *args: Any) -> tuple[int, bytes]:
        return 250, b"ok"

    def starttls(self, *args: Any, **kwargs: Any) -> tuple[int, bytes]:
        self.started_tls = True
        return 220, b"ready"

    def login(self, user: str, password: str) -> tuple[int, bytes]:
        self.logged_in_as = user
        return 235, b"authenticated"

    def send_message(self, msg: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
        if self.controller.fail_with is not None:
            raise self.controller.fail_with
        self.messages.append(msg)
        return {}

    def sendmail(self, from_addr: str, to_addrs: Any, msg: Any, *args: Any, **kw: Any) -> dict:
        return self.send_message(msg)

    def quit(self) -> tuple[int, bytes]:
        self.closed = True
        return 221, b"bye"

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def fake_smtp(monkeypatch: pytest.MonkeyPatch) -> FakeSMTPController:
    controller = FakeSMTPController()
    plain = type("FakeSMTPPlain", (FakeSMTP,), {"controller": controller, "ssl": False})
    secure = type("FakeSMTPSSL", (FakeSMTP,), {"controller": controller, "ssl": True})
    monkeypatch.setattr(smtplib, "SMTP", plain)
    monkeypatch.setattr(smtplib, "SMTP_SSL", secure)
    return controller
