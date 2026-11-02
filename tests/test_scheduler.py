"""NYSE market calendar and the background watchlist scheduler (fake coordinator)."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Iterator
from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from trading_swarm import scheduler as scheduler_module
from trading_swarm.config import Settings
from trading_swarm.logging_setup import request_id_ctx
from trading_swarm.models import ScanReport
from trading_swarm.scheduler import (
    SwarmScheduler,
    WatchlistScanner,
    compute_next_delay,
    is_us_market_open,
    market_session,
    next_market_open,
    nyse_early_closes,
    nyse_holidays,
    to_new_york,
)

MONDAY_OPEN = datetime(2026, 9, 14, 13, 30, tzinfo=UTC)  # 09:30 EDT
FRIDAY_EVENING = datetime(2026, 9, 11, 20, 30, tzinfo=UTC)  # 16:30 EDT, closed
SATURDAY = datetime(2026, 9, 12, 15, 0, tzinfo=UTC)


def utc(*args: int) -> datetime:
    return datetime(*args, tzinfo=UTC)


def wait_until(predicate: Callable[[], bool], timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


class FakeCoordinator:
    """Counts ``scan_watchlist`` calls; can fail the first N calls or block until released."""

    def __init__(self, *, fail_times: int = 0, block: threading.Event | None = None) -> None:
        self.fail_times = fail_times
        self.block = block
        self.started = threading.Event()
        self.request_ids: list[str | None] = []
        self._calls = 0
        self._lock = threading.Lock()

    @property
    def calls(self) -> int:
        with self._lock:
            return self._calls

    def scan_watchlist(self) -> ScanReport:
        with self._lock:
            self._calls += 1
            call = self._calls
        self.request_ids.append(request_id_ctx.get())
        self.started.set()
        if self.block is not None:
            self.block.wait(5)
        if call <= self.fail_times:
            raise RuntimeError(f"provider down (call {call})")
        return ScanReport()


@pytest.fixture
def schedulers() -> Iterator[list[SwarmScheduler]]:
    """Collects schedulers so their threads are always stopped after a test."""
    created: list[SwarmScheduler] = []
    yield created
    for item in created:
        item.stop(timeout=5)


@pytest.fixture
def make_scheduler(
    make_settings: Callable[..., Settings], schedulers: list[SwarmScheduler]
) -> Callable[..., SwarmScheduler]:
    def factory(
        coordinator: WatchlistScanner,
        *,
        interval_seconds: float | None = 0.02,
        run_on_start: bool = True,
        clock: Callable[[], datetime] | None = None,
        **overrides: object,
    ) -> SwarmScheduler:
        settings = make_settings(**{"scheduler_enabled": True, **overrides})
        kwargs = {"clock": clock} if clock is not None else {}
        item = SwarmScheduler(
            coordinator,
            settings,
            interval_seconds=interval_seconds,
            run_on_start=run_on_start,
            **kwargs,
        )
        schedulers.append(item)
        return item

    return factory


# ============================================================================== calendar


@pytest.mark.parametrize(
    ("year", "expected"),
    [
        (
            2025,
            [
                "01-01",
                "01-20",
                "02-17",
                "04-18",
                "05-26",
                "06-19",
                "07-04",
                "09-01",
                "11-27",
                "12-25",
            ],
        ),
        (
            2026,
            [
                "01-01",
                "01-19",
                "02-16",
                "04-03",
                "05-25",
                "06-19",
                "07-03",
                "09-07",
                "11-26",
                "12-25",
            ],
        ),
        (
            # Juneteenth (Sat → Fri), Independence Day (Sun → Mon), Christmas (Sat → Fri).
            2027,
            [
                "01-01",
                "01-18",
                "02-15",
                "03-26",
                "05-31",
                "06-18",
                "07-05",
                "09-06",
                "11-25",
                "12-24",
            ],
        ),
        (
            # New Year's Day on a Saturday is not observed on the preceding Friday.
            2028,
            ["01-17", "02-21", "04-14", "05-29", "06-19", "07-04", "09-04", "11-23", "12-25"],
        ),
    ],
)
def test_nyse_holidays(year: int, expected: list[str]) -> None:
    assert sorted(d.strftime("%m-%d") for d in nyse_holidays(year)) == expected


def test_nyse_early_closes() -> None:
    assert sorted(nyse_early_closes(2025)) == [
        date(2025, 7, 3),
        date(2025, 11, 28),
        date(2025, 12, 24),
    ]
    # 2026: July 3 is itself the (observed) holiday, so only two early closes.
    assert sorted(nyse_early_closes(2026)) == [date(2026, 11, 27), date(2026, 12, 24)]


@pytest.mark.parametrize(
    ("moment", "is_open"),
    [
        (MONDAY_OPEN, True),  # the opening bell is inclusive
        (MONDAY_OPEN - timedelta(minutes=1), False),
        (utc(2026, 9, 14, 19, 59), True),
        (utc(2026, 9, 14, 20, 0), False),  # 16:00 EDT close is exclusive
        (utc(2026, 1, 5, 14, 30), True),  # winter: 09:30 EST is 14:30 UTC
        (utc(2026, 1, 5, 14, 29), False),
        (utc(2026, 1, 5, 13, 45), False),  # would be open in summer
        (utc(2026, 3, 9, 13, 30), True),  # first Monday after DST starts
        (utc(2026, 11, 2, 14, 30), True),  # first Monday after DST ends
        (utc(2026, 11, 2, 13, 45), False),
        (SATURDAY, False),
        (utc(2026, 7, 3, 15, 0), False),  # Independence Day (observed)
        (utc(2026, 4, 3, 15, 0), False),  # Good Friday
        (utc(2026, 11, 27, 17, 59), True),  # day after Thanksgiving: 13:00 EST close
        (utc(2026, 11, 27, 18, 0), False),
        (datetime(2026, 9, 14, 15, 0), True),  # naive = UTC
    ],
)
def test_is_us_market_open(moment: datetime, is_open: bool) -> None:
    assert is_us_market_open(moment) is is_open


@pytest.mark.parametrize(
    ("moment", "expected"),
    [
        (FRIDAY_EVENING, MONDAY_OPEN),
        (utc(2026, 9, 14, 12, 0), MONDAY_OPEN),  # pre-market → same day
        (utc(2026, 9, 14, 15, 0), utc(2026, 9, 15, 13, 30)),  # during the session → tomorrow
        (utc(2026, 11, 25, 22, 0), utc(2026, 11, 27, 14, 30)),  # skips Thanksgiving
        (utc(2026, 12, 24, 19, 0), utc(2026, 12, 28, 14, 30)),  # Christmas + weekend
    ],
)
def test_next_market_open(moment: datetime, expected: datetime) -> None:
    assert next_market_open(moment) == expected


def test_market_session_and_new_york_conversion() -> None:
    assert market_session(date(2026, 9, 12)) is None
    assert market_session(date(2026, 12, 24)) == (
        utc(2026, 12, 24, 14, 30),
        utc(2026, 12, 24, 18, 0),
    )
    assert to_new_york(MONDAY_OPEN).strftime("%H:%M %Z") == "09:30 EDT"
    assert is_us_market_open() in (True, False)  # default "now" works


@pytest.mark.parametrize(
    ("year", "easter"),
    [(2000, (4, 23)), (2024, (3, 31)), (2025, (4, 20)), (2026, (4, 5)), (2038, (4, 25))],
)
def test_easter_algorithm(year: int, easter: tuple[int, int]) -> None:
    assert scheduler_module._easter_sunday(year) == date(year, *easter)


def test_fallback_dst_rules_match_the_iana_database(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without tzdata the built-in US DST rules must give identical answers."""
    zone = ZoneInfo("America/New_York")
    samples = []
    for year in range(2024, 2031):
        start, end = scheduler_module._us_dst_bounds_utc(year)
        for anchor in (start, end):
            samples += [anchor + timedelta(minutes=30 * k) for k in range(-6, 7)]
    sessions = [date(2026, 1, 1) + timedelta(days=d) for d in range(366)]
    expected_local = [s.astimezone(zone).replace(tzinfo=None) for s in samples]
    expected_sessions = [market_session(d) for d in sessions]

    monkeypatch.setattr(scheduler_module, "_new_york_zone", lambda: None)
    assert [to_new_york(s).replace(tzinfo=None) for s in samples] == expected_local
    assert [market_session(d) for d in sessions] == expected_sessions


# ============================================================================== scheduling math


@pytest.mark.parametrize(
    ("elapsed", "expected"),
    [(0, 60), (10, 50), (59.5, 0.5), (60, 60), (150, 30), (-1, 60)],
)
def test_compute_next_delay_is_fixed_rate_without_catch_up(elapsed: float, expected: float) -> None:
    assert compute_next_delay(60, elapsed) == pytest.approx(expected)


def test_compute_next_delay_rejects_bad_interval() -> None:
    with pytest.raises(ValueError):
        compute_next_delay(0, 1)


def test_scheduler_rejects_non_positive_interval(settings: Settings) -> None:
    with pytest.raises(ValueError):
        SwarmScheduler(FakeCoordinator(), settings, interval_seconds=0)
    assert SwarmScheduler(FakeCoordinator(), settings).interval_seconds == 3600.0


# ============================================================================== run_once


def test_run_once_returns_report_and_tags_logs_with_a_scan_id(
    make_scheduler: Callable[..., SwarmScheduler],
) -> None:
    coordinator = FakeCoordinator()
    sched = make_scheduler(coordinator)
    report = sched.run_once()
    assert isinstance(report, ScanReport)
    assert sched.last_report is report and coordinator.calls == 1
    assert coordinator.request_ids[0] is not None and coordinator.request_ids[0].startswith("scan-")
    assert request_id_ctx.get() is None
    status = sched.status()
    assert (status["runs"], status["failures"], status["skipped"]) == (1, 0, 0)
    assert status["running"] is False and status["last_duration_seconds"] is not None


def test_run_once_contains_scan_failures(
    make_scheduler: Callable[..., SwarmScheduler], caplog: pytest.LogCaptureFixture
) -> None:
    sched = make_scheduler(FakeCoordinator(fail_times=2))
    with caplog.at_level(logging.ERROR, logger="trading_swarm.scheduler"):
        assert sched.run_once() is None
        assert sched.run_once() is None
    assert "watchlist scan failed" in caplog.text and "provider down" in caplog.text
    assert sched.last_error == "RuntimeError: provider down (call 2)"
    assert sched.status()["consecutive_failures"] == 2

    assert sched.run_once() is not None
    status = sched.status()
    assert (status["runs"], status["failures"], status["consecutive_failures"]) == (1, 2, 0)
    assert sched.last_error is None


def test_market_hours_only_skips_closed_market_unless_forced(
    make_scheduler: Callable[..., SwarmScheduler],
) -> None:
    coordinator = FakeCoordinator()
    clock_value = [SATURDAY]
    sched = make_scheduler(coordinator, clock=lambda: clock_value[0], market_hours_only=True)
    assert sched.run_once() is None and coordinator.calls == 0
    assert sched.status()["skipped"] == 1 and sched.status()["market_open"] is False
    assert sched.run_once(force=True) is not None and coordinator.calls == 1
    clock_value[0] = MONDAY_OPEN + timedelta(hours=1)
    assert sched.run_once() is not None and coordinator.calls == 2


def test_scans_never_overlap(make_scheduler: Callable[..., SwarmScheduler]) -> None:
    release = threading.Event()
    coordinator = FakeCoordinator(block=release)
    sched = make_scheduler(coordinator)
    worker = threading.Thread(target=sched.run_once)
    worker.start()
    try:
        assert coordinator.started.wait(5)
        assert sched.run_once() is None  # the first scan still holds the lock
        assert sched.status()["skipped"] == 1
    finally:
        release.set()
        worker.join(5)
    assert coordinator.calls == 1


# ============================================================================== background loop


def test_background_loop_runs_repeatedly_and_stops_cleanly(
    make_scheduler: Callable[..., SwarmScheduler],
) -> None:
    coordinator = FakeCoordinator()
    sched = make_scheduler(coordinator)
    sched.start()
    assert sched.is_running
    assert wait_until(lambda: coordinator.calls >= 3)
    assert sched.status()["next_run_at"] is not None
    sched.stop()
    assert not sched.is_running
    calls = coordinator.calls
    time.sleep(0.1)
    assert coordinator.calls == calls  # nothing runs after stop()
    assert sched.status()["next_run_at"] is None


def test_loop_survives_failing_scans(make_scheduler: Callable[..., SwarmScheduler]) -> None:
    coordinator = FakeCoordinator(fail_times=2)
    sched = make_scheduler(coordinator)
    sched.start()
    assert wait_until(lambda: sched.status()["runs"] >= 2)
    sched.stop()
    status = sched.status()
    assert status["failures"] == 2 and status["consecutive_failures"] == 0


def test_start_respects_scheduler_enabled_and_is_idempotent(
    make_scheduler: Callable[..., SwarmScheduler], caplog: pytest.LogCaptureFixture
) -> None:
    disabled = make_scheduler(FakeCoordinator(), scheduler_enabled=False)
    with caplog.at_level(logging.INFO, logger="trading_swarm.scheduler"):
        disabled.start()
    assert not disabled.is_running and "scheduler disabled" in caplog.text
    disabled.start(force=True)
    assert disabled.is_running
    disabled.stop()

    sched = make_scheduler(FakeCoordinator(), interval_seconds=3600, run_on_start=False)
    sched.start()
    first = sched._thread
    sched.start()
    assert sched._thread is first
    assert sum(t.name == "swarm-scheduler" and t.is_alive() for t in threading.enumerate()) == 1


def test_stop_wakes_a_sleeping_loop_immediately(
    make_scheduler: Callable[..., SwarmScheduler],
) -> None:
    coordinator = FakeCoordinator()
    sched = make_scheduler(coordinator, interval_seconds=3600, run_on_start=False)
    sched.stop()  # stopping a never-started scheduler is a no-op
    sched.start()
    began = time.monotonic()
    sched.stop(timeout=5)
    assert time.monotonic() - began < 2.0
    assert not sched.is_running and coordinator.calls == 0


def test_scheduler_can_restart_after_stop(make_scheduler: Callable[..., SwarmScheduler]) -> None:
    coordinator = FakeCoordinator()
    sched = make_scheduler(coordinator)
    sched.start()
    assert wait_until(lambda: coordinator.calls >= 1)
    sched.stop()
    before = coordinator.calls
    sched.start()
    assert wait_until(lambda: coordinator.calls > before)
    sched.stop()
    assert not sched.is_running


def test_stop_waits_for_a_running_scan_to_finish(
    make_scheduler: Callable[..., SwarmScheduler], caplog: pytest.LogCaptureFixture
) -> None:
    release = threading.Event()
    coordinator = FakeCoordinator(block=release)
    sched = make_scheduler(coordinator)
    sched.start()
    assert coordinator.started.wait(5)
    with caplog.at_level(logging.WARNING, logger="trading_swarm.scheduler"):
        sched.stop(timeout=0.05)
    assert sched.is_running and "still busy" in caplog.text
    release.set()
    sched.stop(timeout=5)
    assert not sched.is_running and coordinator.calls == 1


def test_market_hours_loop_sleeps_until_the_opening_bell(
    make_scheduler: Callable[..., SwarmScheduler],
) -> None:
    coordinator = FakeCoordinator()
    sched = make_scheduler(coordinator, clock=lambda: FRIDAY_EVENING, market_hours_only=True)
    sched.start()
    expected = (MONDAY_OPEN + timedelta(seconds=1)).isoformat()
    assert wait_until(lambda: sched.status()["next_run_at"] == expected)
    sched.stop()
    assert coordinator.calls == 0  # no polling while the market is closed


def test_fake_coordinator_satisfies_protocol() -> None:
    assert isinstance(FakeCoordinator(), WatchlistScanner)


# ----------------------------------------------------------------------------- sessions


@pytest.mark.parametrize(
    ("moment", "expected"),
    [
        # Thursday 2026-07-02 after the close → that session.
        (datetime(2026, 7, 2, 21, 0, tzinfo=UTC), date(2026, 7, 2)),
        # Friday 2026-07-03 is the observed Independence Day holiday → still Thursday.
        (datetime(2026, 7, 3, 18, 0, tzinfo=UTC), date(2026, 7, 2)),
        # Saturday and Sunday → Thursday (no session since).
        (datetime(2026, 7, 4, 12, 0, tzinfo=UTC), date(2026, 7, 2)),
        # Monday 2026-07-06 before the 09:30 New York open (13:30 UTC) → Thursday.
        (datetime(2026, 7, 6, 13, 29, tzinfo=UTC), date(2026, 7, 2)),
        # ...and from the opening bell on → Monday.
        (datetime(2026, 7, 6, 13, 30, tzinfo=UTC), date(2026, 7, 6)),
        # 02:00 UTC Tuesday is still Monday evening in New York.
        (datetime(2026, 7, 7, 2, 0, tzinfo=UTC), date(2026, 7, 6)),
    ],
)
def test_session_day(moment: datetime, expected: date) -> None:
    assert scheduler_module.session_day(moment) == expected


def test_trading_days_skip_weekends_and_holidays() -> None:
    days = scheduler_module.trading_days(date(2026, 7, 1), date(2026, 7, 7))
    assert days == [date(2026, 7, 1), date(2026, 7, 2), date(2026, 7, 6), date(2026, 7, 7)]
    assert scheduler_module.trading_days(date(2026, 7, 7), date(2026, 7, 1)) == []
    assert scheduler_module.is_trading_day(date(2026, 12, 25)) is False
