"""API-key authentication and in-process rate limiting.

Authentication
--------------
Clients send ``X-API-Key: <key>``; accepted keys come from ``API_KEYS`` (comma list).
Keys are compared in constant time: both sides are hashed with SHA-256 (equal-length
digests) and compared with :func:`hmac.compare_digest` against *every* configured key,
so neither the comparison nor the number of keys checked leaks which prefix matched.
With ``API_KEYS`` empty, authentication is disabled (the API logs a warning at start-up).

Rate limiting
-------------
A sliding-window log per client: at most ``RATE_LIMIT_PER_MINUTE`` accepted requests
in any 60-second window. Requests over the limit get ``429`` with a ``Retry-After``
header (seconds until the oldest request in the window expires) and are *not* counted,
so a client that honours ``Retry-After`` gets through immediately afterwards.

The client identity is the API key's fingerprint when a *valid* key is presented,
otherwise the client IP. Requests carrying wrong keys are therefore throttled per IP —
the limiter doubles as brute-force protection for the key check, which runs after it.
No public endpoint answers differently for a valid key (``/health`` returns the same
body to everyone), so there is no unthrottled way around it.
Behind a reverse proxy, run uvicorn with ``--proxy-headers --forwarded-allow-ips=…`` so
the client IP is the real one rather than the proxy's.

The limiter is in-process: with several worker processes each enforces its own window
(put a shared limiter in the reverse proxy if that matters).

Cross-site request guard (CSRF)
-------------------------------
CORS only stops a foreign web page from *reading* responses. Some requests count as
"simple": a bodiless ``POST /portfolio/reset``, ``POST /scan`` or ``POST /alerts/test``,
or a ``text/plain`` body. A browser sends those without a preflight, so they would still
*execute*. With authentication off, a malicious page could reset the operator's
localhost account, trigger auto-trading scans or spam alert e-mails. API keys travel in a
header that browsers never attach on their own, so with authentication on such requests
already fail with 401. To cover the auth-off case, a state-changing request (POST, PUT,
PATCH, DELETE) that a browser marks as coming from another site is refused with
``403 forbidden_origin``, unless:

* its ``Origin`` is listed in ``CORS_ORIGINS`` (``*`` disables the guard), or
* it is same-origin (``Origin`` matches the ``Host`` header, as for the ``/docs`` page).

A browser request is recognised by its ``Origin`` header (browsers send it on every
cross-origin POST), or failing that by ``Sec-Fetch-Site: cross-site|same-site``.
Non-browser clients (the dashboard server, ``curl``, scripts) send neither and are not
affected. DNS-rebinding attacks can defeat the same-origin check. Set ``API_KEYS``
whenever the API is reachable from anything but trusted local processes.
"""

from __future__ import annotations

import hashlib
import hmac
import math
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Final
from urllib.parse import urlsplit

from fastapi import Request, Response, Security
from fastapi.security import APIKeyHeader

from trading_swarm.api.errors import ApiError

API_KEY_HEADER: Final = "X-API-Key"
#: Length of the rate-limit window in seconds ("per minute").
RATE_LIMIT_WINDOW_SECONDS: Final = 60.0
#: Upper bound on tracked client identities (least recently seen are evicted first).
MAX_TRACKED_IDENTITIES: Final = 10_000
#: Methods that change state: the only ones subject to the cross-site guard.
UNSAFE_METHODS: Final = frozenset({"POST", "PUT", "PATCH", "DELETE"})
_DEFAULT_PORTS: Final = {"http": "80", "https": "443"}

api_key_scheme = APIKeyHeader(
    name=API_KEY_HEADER,
    scheme_name="ApiKey",
    description="API key from the server's `API_KEYS` setting (not required when it is empty).",
    auto_error=False,
)


def _digest(value: str) -> bytes:
    return hashlib.sha256(value.encode("utf-8")).digest()


def key_fingerprint(key: str) -> str:
    """Short, non-reversible identifier of a key (for rate-limit buckets and logs)."""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]


class ApiKeyVerifier:
    """Constant-time membership test against the configured API keys."""

    def __init__(self, keys: Iterable[str]) -> None:
        self._digests: tuple[bytes, ...] = tuple(_digest(key) for key in keys if key)

    @property
    def enabled(self) -> bool:
        """``True`` when at least one key is configured (authentication required)."""
        return bool(self._digests)

    def __len__(self) -> int:
        return len(self._digests)

    def __repr__(self) -> str:  # never show key material
        return f"ApiKeyVerifier(keys={len(self._digests)})"

    def verify(self, presented: str | None) -> bool:
        """Whether ``presented`` is one of the configured keys (constant time)."""
        if not presented or not self._digests:
            return False
        candidate = _digest(presented)
        matched = False
        for digest in self._digests:  # no short-circuit: always compare against every key
            matched |= hmac.compare_digest(candidate, digest)
        return matched


@dataclass(frozen=True, slots=True)
class RateLimitDecision:
    """Outcome of :meth:`SlidingWindowRateLimiter.hit`."""

    allowed: bool
    limit: int
    remaining: int
    #: Seconds until the next request would be accepted (0 when allowed).
    retry_after: float


class SlidingWindowRateLimiter:
    """Thread-safe sliding-window-log limiter: ``limit`` requests per ``window`` seconds.

    Each identity keeps the timestamps of its accepted requests inside the window (at
    most ``limit`` of them), so memory is bounded by ``limit × max_identities``.

    Args:
        limit: Maximum accepted requests per window (≥ 1).
        window_seconds: Window length.
        clock: Monotonic time source (injectable for tests).
        max_identities: Identities tracked before the least recently seen are evicted.
    """

    def __init__(
        self,
        limit: int,
        window_seconds: float = RATE_LIMIT_WINDOW_SECONDS,
        *,
        clock: Callable[[], float] = time.monotonic,
        max_identities: int = MAX_TRACKED_IDENTITIES,
    ) -> None:
        if limit < 1:
            raise ValueError(f"rate limit must be at least 1, got {limit!r}")
        if not window_seconds > 0:
            raise ValueError(f"window must be positive, got {window_seconds!r}")
        self.limit = int(limit)
        self.window = float(window_seconds)
        self.max_identities = max(1, int(max_identities))
        self._clock = clock
        self._lock = threading.Lock()
        self._hits: OrderedDict[str, deque[float]] = OrderedDict()

    def hit(self, identity: str) -> RateLimitDecision:
        """Record a request from ``identity`` if it is within the limit."""
        now = self._clock()
        horizon = now - self.window
        with self._lock:
            hits = self._hits.get(identity)
            if hits is None:
                hits = deque()
                self._hits[identity] = hits
            else:
                self._hits.move_to_end(identity)
            while hits and hits[0] <= horizon:
                hits.popleft()
            if len(hits) >= self.limit:
                retry_after = max(0.0, hits[0] + self.window - now)
                return RateLimitDecision(False, self.limit, 0, retry_after)
            hits.append(now)
            remaining = self.limit - len(hits)
            self._evict(horizon)
        return RateLimitDecision(True, self.limit, remaining, 0.0)

    def reset(self) -> None:
        """Forget every client (tests, or after changing the limit)."""
        with self._lock:
            self._hits.clear()

    def tracked_identities(self) -> int:
        with self._lock:
            return len(self._hits)

    def _evict(self, horizon: float) -> None:
        # Least recently seen first: drop idle identities, then enforce the size cap.
        while self._hits:
            oldest_identity, oldest_hits = next(iter(self._hits.items()))
            idle = not oldest_hits or oldest_hits[-1] <= horizon
            if idle or len(self._hits) > self.max_identities:
                del self._hits[oldest_identity]
                continue
            break


# =============================================================================== cross-site guard


def _normalise_origin(origin: str) -> str:
    """``scheme://host[:port]`` lower-cased, without a trailing slash or default port."""
    parts = urlsplit(origin.strip().rstrip("/").lower())
    if not parts.scheme or not parts.netloc:
        return origin.strip().rstrip("/").lower()
    host, _, port = parts.netloc.rpartition(":")
    if host and port == _DEFAULT_PORTS.get(parts.scheme):
        return f"{parts.scheme}://{host}"
    return f"{parts.scheme}://{parts.netloc}"


@dataclass(frozen=True, slots=True)
class AllowedOrigins:
    """Browser origins allowed to make state-changing requests (from ``CORS_ORIGINS``)."""

    any: bool = False
    origins: frozenset[str] = frozenset()

    @classmethod
    def from_list(cls, origins: Iterable[str]) -> AllowedOrigins:
        cleaned = {_normalise_origin(o) for o in origins if o and o.strip()}
        return cls(any="*" in cleaned, origins=frozenset(cleaned - {"*"}))

    def allows(self, origin: str) -> bool:
        return self.any or _normalise_origin(origin) in self.origins


def _is_same_origin(origin: str, request: Request) -> bool:
    """``Origin`` names the host the request was sent to (e.g. the API's own /docs page)."""
    parts = urlsplit(_normalise_origin(origin))
    host = request.headers.get("host", "").strip().lower()
    if not parts.netloc or not host:
        return False
    default = _DEFAULT_PORTS.get(parts.scheme)
    if default and host.endswith(f":{default}"):
        host = host[: -len(default) - 1]
    return parts.netloc == host


def cross_site_rejection(request: Request) -> str | None:
    """Why a state-changing browser request from another site is refused, else ``None``.

    See the module docstring ("Cross-site request guard").
    """
    if request.method.upper() not in UNSAFE_METHODS:
        return None
    allowed: AllowedOrigins = getattr(request.app.state, "allowed_origins", AllowedOrigins())
    if allowed.any:
        return None
    origin = request.headers.get("origin")
    if origin is not None:
        if allowed.allows(origin) or (
            origin.strip() != "null" and _is_same_origin(origin, request)
        ):
            return None
        return (
            f"cross-site request from origin {origin.strip()!r} refused: state-changing "
            "browser requests are only accepted from origins listed in CORS_ORIGINS"
        )
    fetch_site = request.headers.get("sec-fetch-site", "").strip().lower()
    if fetch_site in {"cross-site", "same-site"}:
        return (
            f"cross-site browser request refused (Sec-Fetch-Site: {fetch_site}): "
            "state-changing requests must come from an origin listed in CORS_ORIGINS"
        )
    return None


# =============================================================================== dependencies


def _verifier(request: Request) -> ApiKeyVerifier:
    return request.app.state.api_keys  # type: ignore[no-any-return]


def is_authorized(request: Request, api_key: str | None) -> bool:
    """``True`` when authentication is disabled or ``api_key`` is valid."""
    verifier = _verifier(request)
    return not verifier.enabled or verifier.verify(api_key)


def client_identity(request: Request, api_key: str | None, *, authenticated: bool) -> str:
    """Rate-limit bucket: the key's fingerprint when authenticated, else the client IP."""
    if authenticated and api_key:
        return f"key:{key_fingerprint(api_key)}"
    host = request.client.host if request.client is not None else "unknown"
    return f"ip:{host}"


async def enforce_access(
    request: Request,
    response: Response,
    api_key: str | None = Security(api_key_scheme),
) -> None:
    """Router-level dependency for every protected endpoint: rate limit, cross-site
    guard, then API key.

    Raises:
        ApiError: ``429 rate_limited`` (with ``Retry-After``), ``403 forbidden_origin``
            or ``401 unauthorized``.
    """
    verifier = _verifier(request)
    limiter: SlidingWindowRateLimiter = request.app.state.rate_limiter
    authenticated = verifier.enabled and verifier.verify(api_key)

    decision = limiter.hit(client_identity(request, api_key, authenticated=authenticated))
    rate_headers = {
        "X-RateLimit-Limit": str(decision.limit),
        "X-RateLimit-Remaining": str(decision.remaining),
    }
    if not decision.allowed:
        retry_after = max(1, math.ceil(decision.retry_after))
        raise ApiError(
            429,
            "rate_limited",
            f"rate limit of {decision.limit} requests per minute exceeded; "
            f"retry in {retry_after} s",
            headers={**rate_headers, "Retry-After": str(retry_after)},
            extra={"retry_after": retry_after},
        )
    response.headers.update(rate_headers)

    refusal = cross_site_rejection(request)
    if refusal is not None:
        raise ApiError(403, "forbidden_origin", refusal, headers=rate_headers)

    if verifier.enabled and not authenticated:
        message = (
            f"invalid API key in the {API_KEY_HEADER} header"
            if api_key
            else f"missing API key: send it in the {API_KEY_HEADER} header"
        )
        raise ApiError(
            401,
            "unauthorized",
            message,
            headers={"WWW-Authenticate": f'ApiKey header="{API_KEY_HEADER}"'},
        )


__all__ = [
    "API_KEY_HEADER",
    "MAX_TRACKED_IDENTITIES",
    "RATE_LIMIT_WINDOW_SECONDS",
    "UNSAFE_METHODS",
    "AllowedOrigins",
    "ApiKeyVerifier",
    "RateLimitDecision",
    "SlidingWindowRateLimiter",
    "api_key_scheme",
    "client_identity",
    "cross_site_rejection",
    "enforce_access",
    "is_authorized",
    "key_fingerprint",
]
