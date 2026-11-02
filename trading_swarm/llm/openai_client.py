"""Optional OpenAI headline-sentiment client built on the official ``openai`` SDK.

Uses Chat Completions structured outputs (verified against ``openai`` 3.42.1):
``response_format={"type": "json_schema", "json_schema": {"name", "schema", "strict": True}}``
with the same schema and prompt as the Claude client. Sampling parameters are not sent
(reasoning models such as ``gpt-5-mini`` reject them). Same never-raise contract.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import openai
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


def _request_id(obj: Any) -> str | None:
    for attr in ("_request_id", "request_id"):
        value = getattr(obj, attr, None)
        if isinstance(value, str) and value:
            return value
    return None


class OpenAISentimentClient:
    """:class:`~trading_swarm.llm.base.LLMClient` backed by an OpenAI chat model.

    Args:
        settings: Supplies model, timeout, retries and the API key.
        client: Pre-built ``openai.OpenAI`` client or a test double exposing
            ``.chat.completions.create``.
    """

    name = "openai"
    max_completion_tokens: int = 4096

    def __init__(self, settings: Settings, client: Any | None = None) -> None:
        self.model = settings.openai_model
        self.timeout = settings.llm_timeout_seconds
        if client is None:
            client = openai.OpenAI(
                api_key=settings.openai_api_key or None,
                timeout=settings.llm_timeout_seconds,
                max_retries=settings.llm_max_retries,
            )
        self._client = client

    def build_request(self, symbol: str, headlines: Sequence[NewsItem]) -> dict[str, Any]:
        """Keyword arguments for ``client.chat.completions.create``."""
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(symbol, headlines)},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "headline_sentiment",
                    "schema": SENTIMENT_JSON_SCHEMA,
                    "strict": True,
                },
            },
            "max_completion_tokens": self.max_completion_tokens,
            "timeout": self.timeout,
        }

    def judge_sentiment(
        self, symbol: str, headlines: Sequence[NewsItem]
    ) -> SentimentJudgement | None:
        if not headlines:
            return None
        try:
            completion = self._client.chat.completions.create(
                **self.build_request(symbol, headlines)
            )
        except openai.AuthenticationError as exc:
            self._log_api_error("authentication failed (check OPENAI_API_KEY)", symbol, exc)
        except openai.PermissionDeniedError as exc:
            self._log_api_error("permission denied for this key/model", symbol, exc)
        except openai.NotFoundError as exc:
            self._log_api_error(f"model {self.model!r} not found (check OPENAI_MODEL)", symbol, exc)
        except openai.RateLimitError as exc:
            self._log_api_error("rate limited after SDK retries", symbol, exc)
        except openai.BadRequestError as exc:
            self._log_api_error("request rejected as invalid", symbol, exc)
        except openai.APIStatusError as exc:
            self._log_api_error(f"API error (HTTP {exc.status_code})", symbol, exc)
        except openai.APIConnectionError as exc:
            logger.warning("OpenAI sentiment for %s: connection error or timeout: %s", symbol, exc)
        except openai.OpenAIError as exc:
            logger.warning("OpenAI sentiment for %s: SDK error: %s", symbol, exc)
        except Exception:
            logger.exception("OpenAI sentiment for %s: unexpected error", symbol)
        else:
            return self._parse(symbol, completion, n_headlines=min(len(headlines), MAX_HEADLINES))
        return None

    def _parse(
        self, symbol: str, completion: Any, *, n_headlines: int
    ) -> SentimentJudgement | None:
        request_id = _request_id(completion)
        choices = getattr(completion, "choices", None) or []
        if not choices:
            logger.warning(
                "OpenAI sentiment for %s: no choices (request_id=%s)", symbol, request_id
            )
            return None
        choice = choices[0]
        message = getattr(choice, "message", None)
        finish_reason = getattr(choice, "finish_reason", None)
        refusal = getattr(message, "refusal", None)
        if refusal:
            logger.warning(
                "OpenAI declined sentiment for %s (request_id=%s): %s", symbol, request_id, refusal
            )
            return None
        if finish_reason in {"length", "content_filter"}:
            logger.warning(
                "OpenAI sentiment for %s stopped early: finish_reason=%s (request_id=%s)",
                symbol,
                finish_reason,
                request_id,
            )
            return None
        content = getattr(message, "content", None)
        if not content:
            logger.warning(
                "OpenAI sentiment for %s: empty content (request_id=%s)", symbol, request_id
            )
            return None
        try:
            judgement = SentimentJudgement.model_validate_json(content)
        except ValidationError as exc:
            logger.warning(
                "OpenAI sentiment for %s: unparseable output (request_id=%s): %s",
                symbol,
                request_id,
                exc.errors(include_url=False)[:3],
            )
            return None
        return finalize_judgement(judgement, n_headlines)

    def _log_api_error(self, what: str, symbol: str, exc: openai.APIStatusError) -> None:
        logger.warning(
            "OpenAI sentiment for %s: %s (status=%s, request_id=%s): %s",
            symbol,
            what,
            getattr(exc, "status_code", None),
            _request_id(exc),
            getattr(exc, "message", exc),
        )


__all__ = ["OpenAISentimentClient"]
