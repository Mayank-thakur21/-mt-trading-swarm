"""Background watchlist scanner and a small US equity market calendar.

:class:`SwarmScheduler` runs ``coordinator.scan_watchlist()`` on a daemon thread every
``SCAN_INTERVAL_MINUTES`` when ``SCHEDULER_ENABLED=true``:

* **Robust.** A failing scan (provider outage, bug) is logged with its traceback and
  counted; the loop keeps going. Scans never overlap.
* **Clean shutdown.** The loop sleeps on a :class:`threading.Event`, not ``time.sleep``,
  so :meth:`SwarmScheduler.stop` wakes it immediately; it then joins the thread. A scan
  already in progress is allowed to finish (it cannot be interrupted safely mid-order).
* **Fixed rate, no catch-up bursts.** Runs are spaced ``interval`` apart measured from
  their start; when a scan overruns, missed ticks are skipped rather than replayed.
* **Market hours.** With ``MARKET_HOURS_ONLY=true`` scans only run during NYSE regular
  trading hours, and while the market is closed the loop sleeps until the next opening
  bell instead of polling.

Market calendar
---------------
:func:`is_us_market_open` implements NYSE regular hours, 09:30-16:00 America/New_York,
on weekdays, minus the exchange's full-day holidays (New Year's Day, Martin Luther King
Jr. Day, Washington's Birthday, Good Friday, Memorial Day, Juneteenth, Independence Day,
Labor Day, Thanksgiving, Christmas — with the NYSE weekend-observance rules) and with
13:00 early closes on July 3, the day after Thanksgiving and Christmas Eve when those
fall on a trading day. Ad-hoc closures (e.g. national days of mourning) cannot be
predicted and are not modelled. :func:`session_day` maps a moment to the session whose
prices it reflects, and :func:`trading_days` lists the sessions in a date range. ``GET
/equity`` uses both to sample the equity curve once per session. Time-zone conversion
uses :mod:`zoneinfo`; if the IANA
database is unavailable (slim containers without ``tzdata``) the post-2007 US
daylight-saving rules are applied directly.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta, timezone, tzinfo
from datetime import time as dtime
from functools import lru_cache
from typing import Any, Final, Protocol, runtime_checkable
from uuid import uuid4
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from trading_swarm.config import Settings
from trading_swarm.logging_setup import request_id_ctx
from trading_swarm.models import ScanReport, ensure_utc, utc_now

logger = logging.getLogger(__name__)

NEW_YORK: Final = "America/New_York"
REGULAR_OPEN: Final = dtime(9, 30)
REGULAR_CLOSE: Final = dtime(16, 0)
EARLY_CLOSE: Final = dtime(13, 0)

_MONDAY, _THURSDAY, _FRIDAY, _SATURDAY, _SUNDAY = 0, 3, 4, 5, 6
#: Seconds added when sleeping until the opening bell, so the wake-up lands inside the session.
_OPEN_SLACK_SECONDS: Final = 1.0
#: Upper bound when searching for the next session (longest scheduled closure is 4 days).
_MAX_CALENDAR_SEARCH_DAYS: Final = 14


# =============================================================================== time zones


@lru_cache(maxsize=1)
def _new_york_zone() -> tzinfo | None:
    try:
        return ZoneInfo(NEW_YORK)
    except ZoneInfoNotFoundError:
        logger.warning(
            "IANA time-zone data for %s not found (install 'tzdata'); "
            "using built-in US daylight-saving rules",
            NEW_YORK,
        )
        return None


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """The ``n``-th ``weekday`` (Mon=0) of a month; ``n=-1`` is the last one."""
    if n > 0:
        first = date(year, month, 1)
        offset = (weekday - first.weekday()) % 7
        return first + timedelta(days=offset + 7 * (n - 1))
    next_month = date(year + month // 12, month % 12 + 1, 1)
    last = next_month - timedelta(days=1)
    return last - timedelta(days=(last.weekday() - weekday) % 7)


def _us_dst_bounds_utc(year: int) -> tuple[datetime, datetime]:
    """US daylight-saving period (Energy Policy Act 2005, in force since 2007) in UTC.

    Starts the second Sunday of March at 02:00 EST (07:00 UTC) and ends the first Sunday
    of November at 02:00 EDT (06:00 UTC).
    """
    start = datetime.combine(_nth_weekday(year, 3, _SUNDAY, 2), dtime(7), UTC)
    end = datetime.combine(_nth_weekday(year, 11, _SUNDAY, 1), dtime(6), UTC)
    return start, end


def _fallback_offset_utc(moment: datetime) -> timedelta:
    start, end = _us_dst_bounds_utc(moment.year)
    return timedelta(hours=-4) if start <= moment < end else timedelta(hours=-5)


def to_new_york(moment: datetime) -> datetime:
    """Convert ``moment`` (naive = UTC) to New York local time."""
    utc = ensure_utc(moment)
    zone = _new_york_zone()
    if zone is not None:
        return utc.astimezone(zone)
    offset = _fallback_offset_utc(utc)
    return utc.astimezone(timezone(offset, "EDT" if offset == timedelta(hours=-4) else "EST"))


def _new_york_to_utc(day: date, at: dtime) -> datetime:
    """A New York wall-clock time on ``day`` as an aware UTC datetime.

    Only used for session times (09:30-16:00), which never fall in a DST transition gap.
    """
    zone = _new_york_zone()
    if zone is not None:
        return datetime.combine(day, at, zone).astimezone(UTC)
    start, end = _us_dst_bounds_utc(day.year)
    daylight = start.date() <= day < end.date()
    offset = timedelta(hours=-4 if daylight else -5)
    return (datetime.combine(day, at, UTC) - offset).astimezone(UTC)


# =============================================================================== NYSE calendar


def _easter_sunday(year: int) -> date:
    """Gregorian Easter Sunday (anonymous Gregorian / Meeus-Jones-Butcher algorithm)."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    ell = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * ell) // 451
    month, day = divmod(h + ell - 7 * m + 114, 31)
    return date(year, month, day + 1)


def _add_observed(holidays: dict[date, str], day: date, name: str) -> None:
    """Saturday holidays are observed on Friday, Sunday holidays on Monday."""
    if day.weekday() == _SATURDAY:
        holidays[day - timedelta(days=1)] = f"{name} (observed)"
    elif day.weekday() == _SUNDAY:
        holidays[day + timedelta(days=1)] = f"{name} (observed)"
    else:
        holidays[day] = name


@lru_cache(maxsize=64)
def nyse_holidays(year: int) -> dict[date, str]:
    """NYSE full-day closures in ``year`` (current rules; Juneteenth from 2022).

    New Year's Day is the exception to the observance rule: when January 1 falls on a
    Saturday the exchange does *not* close on the preceding Friday (NYSE Rule 7.2).
    """
    holidays: dict[date, str] = {}
    new_year = date(year, 1, 1)
    if new_year.weekday() == _SUNDAY:
        holidays[new_year + timedelta(days=1)] = "New Year's Day (observed)"
    elif new_year.weekday() != _SATURDAY:
        holidays[new_year] = "New Year's Day"
    holidays[_nth_weekday(year, 1, _MONDAY, 3)] = "Martin Luther King Jr. Day"
    holidays[_nth_weekday(year, 2, _MONDAY, 3)] = "Washington's Birthday"
    holidays[_easter_sunday(year) - timedelta(days=2)] = "Good Friday"
    holidays[_nth_weekday(year, 5, _MONDAY, -1)] = "Memorial Day"
    if year >= 2022:
        _add_observed(holidays, date(year, 6, 19), "Juneteenth National Independence Day")
    _add_observed(holidays, date(year, 7, 4), "Independence Day")
    holidays[_nth_weekday(year, 9, _MONDAY, 1)] = "Labor Day"
    holidays[_nth_weekday(year, 11, _THURSDAY, 4)] = "Thanksgiving Day"
    _add_observed(holidays, date(year, 12, 25), "Christmas Day")
    return holidays


@lru_cache(maxsize=64)
def nyse_early_closes(year: int) -> dict[date, str]:
    """Scheduled 13:00 closes: July 3 and Christmas Eve (Mon-Thu only), day after Thanksgiving."""
    closes: dict[date, str] = {}
    july_3 = date(year, 7, 3)
    if july_3.weekday() <= _THURSDAY:
        closes[july_3] = "Independence Day eve"
    closes[_nth_weekday(year, 11, _THURSDAY, 4) + timedelta(days=1)] = "Day after Thanksgiving"
    christmas_eve = date(year, 12, 24)
    if christmas_eve.weekday() <= _THURSDAY:
        closes[christmas_eve] = "Christmas Eve"
    return closes


def market_session(day: date) -> tuple[datetime, datetime] | None:
    """Regular session ``(open, close)`` in UTC for a New York calendar ``day``.

    ``None`` on weekends and NYSE holidays.
    """
    if day.weekday() >= _SATURDAY or day in nyse_holidays(day.year):
        return None
    close = EARLY_CLOSE if day in nyse_early_closes(day.year) else REGULAR_CLOSE
    return _new_york_to_utc(day, REGULAR_OPEN), _new_york_to_utc(day, close)


def is_trading_day(day: date) -> bool:
    """``True`` for NYSE trading days (weekdays that are not full-day holidays)."""
    return day.weekday() < _SATURDAY and day not in nyse_holidays(day.year)


@lru_cache(maxsize=4096)
def _session_open_utc(day: date) -> datetime | None:
    session = market_session(day)
    return session[0] if session is not None else None


def session_day(moment: datetime) -> date:
    """The NYSE session whose prices are the latest ones available at ``moment``.

    This is the most recent trading day whose opening bell rang at or before ``moment``.
    A snapshot taken on a Saturday, or on a Monday before 09:30 New York time, reflects
    Friday's session. A snapshot during or after Monday's session reflects Monday. Naive
    ``moment`` is interpreted as UTC.
    """
    moment = ensure_utc(moment)
    day = to_new_york(moment).date()
    for _ in range(_MAX_CALENDAR_SEARCH_DAYS):
        opened = _session_open_utc(day)
        if opened is not None and opened <= moment:
            return day
        day -= timedelta(days=1)
    raise RuntimeError(f"no NYSE session within {_MAX_CALENDAR_SEARCH_DAYS} days of {moment}")


def trading_days(start: date, end: date) -> list[date]:
    """Every NYSE trading day from ``start`` to ``end`` inclusive (empty if start > end)."""
    days: list[date] = []
    day = start
    while day <= end:
        if is_trading_day(day):
            days.append(day)
        day += timedelta(days=1)
    return days


def is_us_market_open(now: datetime | None = None) -> bool:
    """``True`` during NYSE regular hours (weekdays 09:30-16:00 New York time, holidays
    and early closes respected). Naive ``now`` is interpreted as UTC."""
    moment = ensure_utc(now) if now is not None else utc_now()
    session = market_session(to_new_york(moment).date())
    return session is not None and session[0] <= moment < session[1]


def next_market_open(now: datetime | None = None) -> datetime:
    """The next opening bell strictly after ``now`` (UTC).

    While the market is open this is the *following* session's open.
    """
    moment = ensure_utc(now) if now is not None else utc_now()
    day = to_new_york(moment).date()
    for _ in range(_MAX_CALENDAR_SEARCH_DAYS):
        session = market_session(day)
        if session is not None and moment < session[0]:
            return session[0]
        day += timedelta(days=1)
    raise RuntimeError(f"no NYSE session within {_MAX_CALENDAR_SEARCH_DAYS} days of {moment}")


# =============================================================================== scheduling


def compute_next_delay(interval: float, elapsed: float) -> float:
    """Seconds to wait after a run that took ``elapsed`` seconds (fixed-rate schedule).

    Runs start on a grid ``interval`` apart; a run that overruns skips the ticks it
    missed instead of triggering back-to-back catch-up runs.
    """
    if interval <= 0:
        raise ValueError("interval must be positive")
    elapsed = max(0.0, elapsed)
    if elapsed < interval:
        return interval - elapsed
    return interval - (elapsed % interval)


@runtime_checkable
class WatchlistScanner(Protocol):
    """Anything that can scan the watchlist (the :class:`~trading_swarm.swarm.SwarmCoordinator`)."""

    def scan_watchlist(self) -> ScanReport: ...


def _describe_interval(seconds: float) -> str:
    if seconds >= 60 and seconds % 60 == 0:
        minutes = int(seconds // 60)
        return f"{minutes} min" if minutes < 60 or minutes % 60 else f"{minutes // 60} h"
    return f"{seconds:g} s"


class SwarmScheduler:
    """Runs periodic watchlist scans on a daemon thread.

    Args:
        coordinator: Object with ``scan_watchlist() -> ScanReport``.
        settings: ``scheduler_enabled``, ``scan_interval_minutes``, ``market_hours_only``.
        interval_seconds: Override the interval (tests, sub-minute demos).
        run_on_start: Scan as soon as the thread starts (default) rather than after one
            interval. With ``market_hours_only`` and a closed market the first scan
            happens at the next opening bell either way.
        clock: Source of "now" (UTC) for market-hours decisions; injectable for tests.
    """

    def __init__(
        self,
        coordinator: WatchlistScanner,
        settings: Settings,
        *,
        interval_seconds: float | None = None,
        run_on_start: bool = True,
        clock: Callable[[], datetime] = utc_now,
    ) -> None:
        interval = (
            float(interval_seconds)
            if interval_seconds is not None
            else settings.scan_interval_minutes * 60.0
        )
        if not interval > 0:
            raise ValueError(f"scan interval must be positive, got {interval!r}")
        self.coordinator = coordinator
        self.settings = settings
        self.run_on_start = run_on_start
        self._interval = interval
        self._clock = clock

        self._lifecycle_lock = threading.Lock()
        self._scan_lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        self._last_report: ScanReport | None = None
        self._last_error: str | None = None
        self._last_started_at: datetime | None = None
        self._last_finished_at: datetime | None = None
        self._last_duration: float | None = None
        self._next_run_at: datetime | None = None
        self._runs = 0
        self._failures = 0
        self._consecutive_failures = 0
        self._skipped = 0

    # ------------------------------------------------------------------ state
    @property
    def interval_seconds(self) -> float:
        return self._interval

    @property
    def is_running(self) -> bool:
        """Whether the background thread is alive."""
        thread = self._thread
        return thread is not None and thread.is_alive()

    @property
    def last_report(self) -> ScanReport | None:
        """The most recent successful scan's report."""
        with self._stats_lock:
            return self._last_report

    @property
    def last_error(self) -> str | None:
        """``"ExcType: message"`` of the most recent failed scan (cleared on success)."""
        with self._stats_lock:
            return self._last_error

    def status(self) -> dict[str, Any]:
        """JSON-friendly scheduler state (for ``/health`` or a status endpoint)."""

        def iso(value: datetime | None) -> str | None:
            return value.isoformat() if value is not None else None

        now = self._clock()
        with self._stats_lock:
            return {
                "enabled": self.settings.scheduler_enabled,
                "running": self.is_running,
                "interval_seconds": self._interval,
                "market_hours_only": self.settings.market_hours_only,
                "market_open": is_us_market_open(now),
                "next_run_at": iso(self._next_run_at) if self.is_running else None,
                "last_started_at": iso(self._last_started_at),
                "last_finished_at": iso(self._last_finished_at),
                "last_duration_seconds": self._last_duration,
                "last_error": self._last_error,
                "runs": self._runs,
                "failures": self._failures,
                "consecutive_failures": self._consecutive_failures,
                "skipped": self._skipped,
            }

    # ------------------------------------------------------------------ lifecycle
    def start(self, *, force: bool = False) -> None:
        """Start the background loop (idempotent).

        Does nothing unless ``SCHEDULER_ENABLED=true`` (or ``force=True``), so callers can
        invoke it unconditionally at application start-up.
        """
        with self._lifecycle_lock:
            if not (self.settings.scheduler_enabled or force):
                logger.info("scheduler disabled (SCHEDULER_ENABLED=false); not starting")
                return
            if self.is_running:
                logger.debug("scheduler already running")
                return
            # A fresh Event per thread: a previous thread still finishing its last scan
            # keeps its own (set) event and exits, instead of seeing a cleared one.
            stop_event = threading.Event()
            thread = threading.Thread(
                target=self._loop, args=(stop_event,), name="swarm-scheduler", daemon=True
            )
            self._stop_event = stop_event
            self._thread = thread
            thread.start()
        logger.info(
            "scheduler started: watchlist scan every %s%s",
            _describe_interval(self._interval),
            " during US market hours" if self.settings.market_hours_only else "",
        )

    def stop(self, timeout: float = 10.0) -> None:
        """Signal the loop to stop and join the thread (waits for a running scan)."""
        with self._lifecycle_lock:
            thread = self._thread
            self._stop_event.set()
        if thread is None:
            return
        if thread is threading.current_thread():
            return  # called from within a scan: the loop exits after it returns
        thread.join(timeout)
        if thread.is_alive():
            logger.warning(
                "scheduler thread still busy after %.1fs (a scan is running); "
                "it will exit when the scan finishes",
                timeout,
            )
            return
        with self._lifecycle_lock:
            if self._thread is thread:
                self._thread = None
        with self._stats_lock:
            self._next_run_at = None
        logger.info("scheduler stopped")

    # ------------------------------------------------------------------ scanning
    def run_once(self, *, force: bool = False) -> ScanReport | None:
        """Run one scan now.

        Returns ``None`` when the scan was skipped (market closed with
        ``MARKET_HOURS_ONLY``, unless ``force``; or another scan is still running) or
        failed (the exception is logged and recorded in :meth:`status`). Never raises
        :class:`Exception`.
        """
        if self.settings.market_hours_only and not force:
            now = self._clock()
            if not is_us_market_open(now):
                with self._stats_lock:
                    self._skipped += 1
                logger.info(
                    "US market closed; skipping scheduled scan (next open %s)",
                    next_market_open(now).isoformat(),
                )
                return None
        if not self._scan_lock.acquire(blocking=False):
            with self._stats_lock:
                self._skipped += 1
            logger.warning("previous watchlist scan still running; skipping this one")
            return None
        token = request_id_ctx.set(f"scan-{uuid4().hex[:8]}")
        try:
            return self._scan()
        finally:
            request_id_ctx.reset(token)
            self._scan_lock.release()

    def _scan(self) -> ScanReport | None:
        started_at = self._clock()
        t0 = time.monotonic()
        with self._stats_lock:
            self._last_started_at = started_at
        logger.info("watchlist scan started")
        try:
            report = self.coordinator.scan_watchlist()
        except Exception as exc:
            duration = time.monotonic() - t0
            with self._stats_lock:
                self._failures += 1
                self._consecutive_failures += 1
                self._last_error = f"{type(exc).__name__}: {exc}"
                self._last_finished_at = self._clock()
                self._last_duration = duration
                failures = self._consecutive_failures
            logger.exception(
                "watchlist scan failed after %.1fs (%d consecutive failure(s)); "
                "will retry at the next interval",
                duration,
                failures,
            )
            return None
        duration = time.monotonic() - t0
        with self._stats_lock:
            self._runs += 1
            self._consecutive_failures = 0
            self._last_error = None
            self._last_report = report
            self._last_finished_at = self._clock()
            self._last_duration = duration
        logger.info(
            "watchlist scan finished in %.1fs: %d decision(s), %d exit(s), %d error(s)",
            duration,
            len(getattr(report, "decisions", None) or []),
            len(getattr(report, "exits", None) or []),
            len(getattr(report, "errors", None) or []),
        )
        return report

    # ------------------------------------------------------------------ loop
    def _loop(self, stop_event: threading.Event) -> None:
        delay = self._market_adjusted(0.0 if self.run_on_start else self._interval)
        while True:
            with self._stats_lock:
                self._next_run_at = self._clock() + timedelta(seconds=delay)
            if stop_event.wait(delay):
                break
            started = time.monotonic()
            try:
                self.run_once()
            except Exception:  # pragma: no cover - run_once already contains scan errors
                logger.exception("unexpected scheduler error; continuing")
            if stop_event.is_set():
                break
            elapsed = time.monotonic() - started
            if elapsed > self._interval:
                logger.warning(
                    "scan took %.0fs, longer than the %s interval; skipping %d missed run(s)",
                    elapsed,
                    _describe_interval(self._interval),
                    int(elapsed // self._interval),
                )
            delay = self._market_adjusted(compute_next_delay(self._interval, elapsed))
        logger.debug("scheduler loop exited")

    def _market_adjusted(self, delay: float) -> float:
        """With ``market_hours_only``, push a wake-up that lands in a closed market to the
        next opening bell (plus a small slack)."""
        if not self.settings.market_hours_only:
            return delay
        try:
            now = self._clock()
            wake = now + timedelta(seconds=delay)
            if is_us_market_open(wake):
                return delay
            until_open = (next_market_open(wake) - now).total_seconds() + _OPEN_SLACK_SECONDS
            logger.debug("market closed at next tick; sleeping %.0fs until the open", until_open)
            return max(delay, until_open)
        except Exception:  # pragma: no cover - calendar bug must not kill the loop
            logger.exception("market-hours calculation failed; using the plain interval")
            return delay


__all__ = [
    "EARLY_CLOSE",
    "NEW_YORK",
    "REGULAR_CLOSE",
    "REGULAR_OPEN",
    "SwarmScheduler",
    "WatchlistScanner",
    "compute_next_delay",
    "is_trading_day",
    "is_us_market_open",
    "market_session",
    "next_market_open",
    "nyse_early_closes",
    "nyse_holidays",
    "session_day",
    "to_new_york",
    "trading_days",
]
