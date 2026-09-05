"""Activity session tracking.

Segments the timeline into "you were using the machine" runs so the app can
report real screen time and prompt for breaks.

Honesty note: activity here is inferred from *resource movement*, not from
keyboard or mouse input. Reading real input-idle time needs OS-specific APIs
(and on Linux, an X11/Wayland dependency), which this project deliberately
avoids. The consequence is that a long compile with nobody at the desk counts as
active, and reading a static page with no CPU load may not. The UI labels this
"activity", never "presence".
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from .collector import Sample


@dataclass
class SessionState:
    """Snapshot of the current activity session for the UI."""

    active: bool = False
    session_start: float | None = None
    active_seconds: float = 0.0
    idle_seconds: float = 0.0
    continuous_minutes: float = 0.0
    break_due: bool = False
    session_count_today: int = 0
    active_seconds_today: float = 0.0

    @property
    def session_duration(self) -> float:
        if self.session_start is None:
            return 0.0
        return max(0.0, time.time() - self.session_start)


class SessionTracker:
    """State machine that opens, extends and closes activity sessions."""

    def __init__(
        self,
        storage,
        cpu_threshold: float = 12.0,
        io_threshold: float = 0.6,
        idle_grace_seconds: float = 120.0,
        break_reminder_minutes: float = 55.0,
        sample_interval: float = 1.0,
    ) -> None:
        self.storage = storage
        self.cpu_threshold = cpu_threshold
        self.io_threshold = io_threshold
        self.idle_grace_seconds = idle_grace_seconds
        self.break_reminder_minutes = break_reminder_minutes
        self.sample_interval = sample_interval

        self._session_id: int | None = None
        self._session_start: float | None = None
        self._active_seconds = 0.0
        self._calm_seconds = 0.0
        self._samples = 0
        self._cpu_total = 0.0
        self._ram_total = 0.0
        self._break_prompted_at: float | None = None
        self._last_flush = 0.0
        self._day = time.localtime().tm_yday
        self._sessions_today = 0
        self._active_today = 0.0

        # A previous run may have been killed mid-session.
        try:
            self.storage.close_dangling_sessions()
        except Exception:
            pass

    def is_active(self, sample: Sample) -> bool:
        """Whether this sample looks like the machine is being worked on."""
        if sample.cpu >= self.cpu_threshold:
            return True
        if sample.disk_io_mbps >= self.io_threshold:
            return True
        if sample.net_io_mbps >= self.io_threshold:
            return True
        return False

    def _roll_day(self) -> None:
        today = time.localtime().tm_yday
        if today != self._day:
            self._day = today
            self._sessions_today = 0
            self._active_today = 0.0

    def observe(self, sample: Sample) -> tuple[SessionState, list[dict]]:
        """Advance the state machine. Returns (state, events_to_log)."""
        self._roll_day()
        events: list[dict] = []
        active = self.is_active(sample)
        step = self.sample_interval

        if active:
            self._calm_seconds = 0.0
            if self._session_id is None:
                self._session_start = sample.ts
                self._active_seconds = 0.0
                self._samples = 0
                self._cpu_total = 0.0
                self._ram_total = 0.0
                self._break_prompted_at = None
                try:
                    self._session_id = self.storage.open_session(sample.ts)
                except Exception:
                    self._session_id = -1
                self._sessions_today += 1
            self._active_seconds += step
            self._active_today += step
            self._samples += 1
            self._cpu_total += sample.cpu
            self._ram_total += sample.ram
        else:
            if self._session_id is not None:
                self._calm_seconds += step
                if self._calm_seconds >= self.idle_grace_seconds:
                    events.extend(self._close(sample.ts))

        # Persist progress periodically rather than every second.
        if self._session_id not in (None, -1) and sample.ts - self._last_flush >= 15:
            self._flush(sample.ts)
            self._last_flush = sample.ts

        continuous_minutes = 0.0
        if self._session_start is not None and self._session_id is not None:
            continuous_minutes = (sample.ts - self._session_start) / 60.0

        break_due = False
        if (
            self._session_id is not None
            and continuous_minutes >= self.break_reminder_minutes
        ):
            break_due = True
            # Re-prompt at most once per reminder interval.
            if (
                self._break_prompted_at is None
                or sample.ts - self._break_prompted_at >= self.break_reminder_minutes * 60
            ):
                self._break_prompted_at = sample.ts
                events.append(
                    {
                        "kind": "wellbeing",
                        "severity": "info",
                        "key": "break_reminder",
                        "message": (
                            f"You have been working continuously for "
                            f"{continuous_minutes:.0f} minutes. Consider a short break."
                        ),
                        "value": continuous_minutes,
                    }
                )

        state = SessionState(
            active=active,
            session_start=self._session_start if self._session_id is not None else None,
            active_seconds=self._active_seconds,
            idle_seconds=self._calm_seconds,
            continuous_minutes=continuous_minutes,
            break_due=break_due,
            session_count_today=self._sessions_today,
            active_seconds_today=self._active_today,
        )
        return state, events

    def _flush(self, now: float, closed: bool = False) -> None:
        if self._session_id in (None, -1):
            return
        cpu_avg = self._cpu_total / self._samples if self._samples else 0.0
        ram_avg = self._ram_total / self._samples if self._samples else 0.0
        try:
            self.storage.update_session(
                self._session_id,
                end_ts=now,
                active_seconds=self._active_seconds,
                cpu_avg=cpu_avg,
                ram_avg=ram_avg,
                samples=self._samples,
                closed=closed,
            )
        except Exception:
            pass

    def _close(self, now: float) -> list[dict]:
        minutes = self._active_seconds / 60.0
        self._flush(now, closed=True)
        events = [
            {
                "kind": "session",
                "severity": "info",
                "key": "session_end",
                "message": f"Activity session ended after {minutes:.0f} active minutes.",
                "value": minutes,
            }
        ]
        self._session_id = None
        self._session_start = None
        self._active_seconds = 0.0
        self._calm_seconds = 0.0
        self._samples = 0
        self._break_prompted_at = None
        return events

    def shutdown(self) -> None:
        if self._session_id not in (None, -1):
            self._flush(time.time(), closed=True)
