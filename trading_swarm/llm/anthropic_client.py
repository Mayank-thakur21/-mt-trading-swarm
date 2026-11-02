"""Claude headline-sentiment client built on the official ``anthropic`` SDK (default provider).

Request shape (verified against ``anthropic`` 2.3.0, ``client.beta.messages.create``):

* ``output_config.format`` = JSON schema → the response text is guaranteed to be JSON that
  matches :data:`~trading_swarm.llm.prompts.SENTIMENT_JSON_SCHEMA`;
* ``output_config.effort`` (default ``"low"``) is the cost/latency control — the default model
  always thinks and it cannot be disabled, so no ``thinking`` parameter is sent, and sampling
  parameters (``temperature``/``top_p``/``top_k``) are not sent because current models
  reject them;
* ``betas=["server-side-fallback-2026-07-01"]`` + ``fallbacks="default"``: if the model
  declines for policy reasons the API retries server-side on its default fallback model.

The SDK itself retries 408/409/429/5xx and connection errors ``LLM_MAX_RETRIES`` times.
Everything else is caught here and turned into ``None`` (see :mod:`trading_swarm.llm.base`).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any, Final

import anthropic
from pydantic import ValidationError

from trading_swarm.config import Settings
from trading_swarm.llm.prompts import (
    MAX_HEADLINES,
    SENTIMENT_JSON_SCHEMA,
    SYSTEM_PROMPT,
    build_user_prompt,
    finalize_judgement,
)
from trading_swarm.models import NewsItem, SentimentJudgement

logger = logging.getLogger(__name__)

FALLBACK_BETA: Final = "server-side-fallback-2026-07-01"


def _request_id(obj: Any) -> str | None:
    """Request id from an SDK response (``_request_id``) or API error (``request_id``)."""
    for attr in ("_request_id", "request_id"):
        value = getattr(obj, attr, None)
        if isinstance(value, str) and value:
            return value
    return None


class AnthropicSentimentClient:
    """:class:`~trading_swarm.llm.base.LLMClient` backed by Claude.

    Args:
        settings: Supplies model, effort, timeout, retries and the API key.
        client: Pre-built SDK client (or a test double exposing
            ``.beta.messages.create``). Built from ``settings`` when omitted.
    """

    name = "anthropic"
    #: Output budget; generous for ~20 headlines of JSON plus low-effort thinking.
    max_tokens: int = 2048

    def __init__(self, settings: Settings, client: Any | None = None) -> None:
        self.model = settings.anthropic_model
        self.effort = settings.anthropic_effort
        if client is None:
            client = anthropic.Anthropic(
                api_key=settings.anthropic_api_key or None,
                timeout=settings.llm_timeout_seconds,
                max_retries=settings.llm_max_retries,
            )
        self._client = client

    def build_request(self, symbol: str, headlines: Sequence[NewsItem]) -> dict[str, Any]:
        """Keyword arguments for ``client.beta.messages.create`` (exposed for tests/logging)."""
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "betas": [FALLBACK_BETA],
            "fallbacks": "default",
            "output_config": {
                "effort": self.effort,
                "format": {"type": "json_schema", "schema": SENTIMENT_JSON_SCHEMA},
            },
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": build_user_prompt(symbol, headlines)}],
        }

    def judge_sentiment(
        self, symbol: str, headlines: Sequence[NewsItem]
    ) -> SentimentJudgement | None:
        if not headlines:
            return None
        request = self.build_request(symbol, headlines)
        try:
            response = self._client.beta.messages.create(**request)
        except anthropic.AuthenticationError as exc:
            self._log_api_error("authentication failed (check ANTHROPIC_API_KEY)", symbol, exc)
        except anthropic.PermissionDeniedError as exc:
            self._log_api_error("permission denied for this key/model", symbol, exc)
        except anthropic.NotFoundError as exc:
            self._log_api_error(
                f"model {self.model!r} not found (check ANTHROPIC_MODEL)", symbol, exc
            )
        except anthropic.RateLimitError as exc:
            self._log_api_error("rate limited after SDK retries", symbol, exc)
        except anthropic.BadRequestError as exc:
            self._log_api_error("request rejected as invalid", symbol, exc)
        except anthropic.APIStatusError as exc:
            self._log_api_error(f"API error (HTTP {exc.status_code})", symbol, exc)
        except anthropic.APIConnectionError as exc:
            logger.warning("Claude sentiment for %s: connection error or timeout: %s", symbol, exc)
        except anthropic.AnthropicError as exc:
            logger.warning("Claude sentiment for %s: SDK error: %s", symbol, exc)
        except Exception:
            logger.exception("Claude sentiment for %s: unexpected error", symbol)
        else:
            return self._parse(symbol, response, n_headlines=min(len(headlines), MAX_HEADLINES))
        return None

    # ------------------------------------------------------------------ internals
    def _parse(self, symbol: str, response: Any, *, n_headlines: int) -> SentimentJudgement | None:
        request_id = _request_id(response)
        stop_reason = getattr(response, "stop_reason", None)
        if stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            logger.warning(
                "Claude declined sentiment for %s (request_id=%s, category=%s, explanation=%s)",
                symbol,
                request_id,
                getattr(details, "category", None),
                getattr(details, "explanation", None),
            )
            return None
        if stop_reason == "max_tokens":
            logger.warning(
                "Claude sentiment for %s hit max_tokens=%d (request_id=%s)",
                symbol,
                self.max_tokens,
                request_id,
            )
            return None
        text = next(
            (
                block.text
                for block in getattr(response, "content", None) or []
                if getattr(block, "type", None) == "text"
            ),
            None,
        )
        if not text:
            logger.warning(
                "Claude sentiment for %s returned no text block (stop_reason=%s, request_id=%s)",
                symbol,
                stop_reason,
                request_id,
            )
            return None
        try:
            judgement = SentimentJudgement.model_validate_json(text)
        except ValidationError as exc:
            logger.warning(
                "Claude sentiment for %s: unparseable output (request_id=%s): %s",
                symbol,
                request_id,
                exc.errors(include_url=False)[:3],
            )
            return None
        usage = getattr(response, "usage", None)
        served_by = getattr(response, "model", None) or self.model
        if served_by != self.model:
            # The requested model declined and the server-side fallback answered.
            logger.info(
                "Claude sentiment for %s served by fallback model %s (request_id=%s)",
                symbol,
                served_by,
                request_id,
            )
        logger.debug(
            "Claude sentiment for %s ok (request_id=%s, model=%s, in=%s, out=%s)",
            symbol,
            request_id,
            served_by,
            getattr(usage, "input_tokens", None),
            getattr(usage, "output_tokens", None),
        )
        return finalize_judgement(judgement, n_headlines)

    def _log_api_error(self, what: str, symbol: str, exc: anthropic.APIStatusError) -> None:
        logger.warning(
            "Claude sentiment for %s: %s (status=%s, request_id=%s): %s",
            symbol,
            what,
            getattr(exc, "status_code", None),
            _request_id(exc),
            getattr(exc, "message", exc),
        )


__all__ = ["FALLBACK_BETA", "AnthropicSentimentClient"]
