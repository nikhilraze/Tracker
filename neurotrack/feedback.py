"""Turning numbers into advice.

Three properties make this useful rather than annoying:

* **Hysteresis** - a rule fires only after the condition holds for several
  consecutive samples, and clears at a *lower* threshold than it triggers at. At
  1 Hz a naive `cpu > 85` check would fire dozens of times a minute.
* **Cooldown** - the same alert is not re-announced for `alert_cooldown` seconds.
* **Attribution** - advice names the processes actually responsible, because
  "CPU is high" is not actionable but "chrome is using 240% CPU" is.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from .collector import ProcessInfo, Sample

SEVERITY_ORDER = {"critical": 0, "warning": 1, "info": 2}


@dataclass
class Alert:
    key: str
    severity: str
    title: str
    detail: str
    value: float | None = None

    @property
    def rank(self) -> int:
        return SEVERITY_ORDER.get(self.severity, 3)


@dataclass
class Advice:
    text: str
    severity: str = "info"
    because: str | None = None

    @property
    def rank(self) -> int:
        return SEVERITY_ORDER.get(self.severity, 3)


@dataclass
class FeedbackReport:
    alerts: list[Alert] = field(default_factory=list)
    advice: list[Advice] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    ram_exhaustion_minutes: float | None = None
    disk_exhaustion_hours: float | None = None

    @property
    def worst_severity(self) -> str | None:
        if not self.alerts:
            return None
        return min(self.alerts, key=lambda a: a.rank).severity


@dataclass
class _RuleState:
    consecutive: int = 0
    firing: bool = False
    last_fired: float = 0.0


@dataclass
class ThresholdRule:
    """A single hysteretic threshold check."""

    key: str
    title: str
    severity: str
    high: float
    clear: float
    getter: Callable[[Sample], float | None]
    unit: str = "%"
    detail_fmt: str = "{value:.0f}{unit} (threshold {high:.0f}{unit})"


def project_exhaustion(
    values: Sequence[float],
    ceiling: float,
    step_seconds: float = 1.0,
    min_points: int = 60,
) -> float | None:
    """Seconds until a rising series would reach ``ceiling``, or None.

    Fits a least-squares line to the recent window. This is the memory-leak
    detector: a steady upward slope in RAM is exactly the signature of a process
    that never releases memory, and unlike the forecast model this is a
    deliberately simple, explainable extrapolation.
    """
    series = np.asarray([v for v in values if v is not None], dtype=float)
    if series.size < min_points:
        return None
    current = float(series[-1])
    if current >= ceiling:
        return 0.0

    x = np.arange(series.size, dtype=float)
    try:
        slope, _intercept = np.polyfit(x, series, 1)
    except Exception:
        return None
    slope_per_second = float(slope) / max(step_seconds, 1e-6)
    # Require a slope that is meaningful, not numerical noise.
    if slope_per_second <= 1e-5:
        return None
    return (ceiling - current) / slope_per_second


class FeedbackEngine:
    """Evaluates rules against the latest sample and emits alerts + advice."""

    def __init__(self, config) -> None:
        self.config = config
        self._states: dict[str, _RuleState] = {}
        self.rules: list[ThresholdRule] = [
            ThresholdRule(
                key="cpu_high",
                title="Sustained high CPU",
                severity="warning",
                high=config.cpu_high,
                clear=config.cpu_clear,
                getter=lambda s: s.cpu,
            ),
            ThresholdRule(
                key="ram_high",
                title="Memory pressure",
                severity="warning",
                high=config.ram_high,
                clear=config.ram_clear,
                getter=lambda s: s.ram,
            ),
            ThresholdRule(
                key="swap_high",
                title="System is swapping",
                severity="critical",
                high=config.swap_high,
                clear=config.swap_clear,
                getter=lambda s: s.swap,
            ),
            ThresholdRule(
                key="disk_high",
                title="Disk almost full",
                severity="critical",
                high=config.disk_high,
                clear=config.disk_clear,
                getter=lambda s: s.disk_pct,
            ),
            ThresholdRule(
                key="temp_high",
                title="Running hot",
                severity="warning",
                high=config.temp_high,
                clear=config.temp_clear,
                getter=lambda s: s.temp_c,
                unit="degC",
            ),
        ]

    def _state(self, key: str) -> _RuleState:
        return self._states.setdefault(key, _RuleState())

    def _should_announce(self, key: str, now: float) -> bool:
        state = self._state(key)
        if now - state.last_fired < self.config.alert_cooldown:
            return False
        state.last_fired = now
        return True

    # ------------------------------------------------------------------
    def evaluate(
        self,
        sample: Sample,
        history: Sequence[Sample],
        forecasts: dict,
        anomaly,
        processes: Sequence[ProcessInfo],
        session_state,
        health: int,
    ) -> FeedbackReport:
        now = sample.ts
        report = FeedbackReport()
        trigger_n = max(1, self.config.trigger_samples)

        top_cpu = [p for p in processes if p.cpu_percent > 5][:3]
        top_mem = sorted(processes, key=lambda p: p.memory_mb, reverse=True)[:3]

        # -- hysteretic threshold rules --------------------------------
        for rule in self.rules:
            value = rule.getter(sample)
            if value is None:
                continue
            state = self._state(rule.key)

            if value >= rule.high:
                state.consecutive += 1
            elif value <= rule.clear:
                # Clearing at a lower threshold prevents flapping around `high`.
                state.consecutive = 0
                if state.firing:
                    state.firing = False
                    report.events.append(
                        {
                            "kind": "alert_clear",
                            "severity": "info",
                            "key": rule.key,
                            "message": f"{rule.title}: recovered ({value:.0f}{rule.unit}).",
                            "value": float(value),
                        }
                    )
            else:
                # Between clear and high: hold current state, do not accumulate.
                state.consecutive = 0

            if state.consecutive >= trigger_n:
                detail = rule.detail_fmt.format(value=value, unit=rule.unit, high=rule.high)
                report.alerts.append(
                    Alert(rule.key, rule.severity, rule.title, detail, float(value))
                )
                if not state.firing:
                    state.firing = True
                if self._should_announce(rule.key, now):
                    report.events.append(
                        {
                            "kind": "alert",
                            "severity": rule.severity,
                            "key": rule.key,
                            "message": f"{rule.title}: {detail}",
                            "value": float(value),
                        }
                    )

        # -- battery ---------------------------------------------------
        if (
            sample.battery_pct is not None
            and sample.battery_plugged is False
            and sample.battery_pct <= self.config.battery_low
        ):
            detail = f"{sample.battery_pct:.0f}% remaining on battery"
            if sample.battery_secs_left:
                detail += f" (~{sample.battery_secs_left // 60} min)"
            report.alerts.append(
                Alert("battery_low", "warning", "Battery low", detail, sample.battery_pct)
            )
            if self._should_announce("battery_low", now):
                report.events.append(
                    {
                        "kind": "alert",
                        "severity": "warning",
                        "key": "battery_low",
                        "message": f"Battery low: {detail}",
                        "value": float(sample.battery_pct),
                    }
                )

        # -- anomaly ---------------------------------------------------
        if anomaly is not None and anomaly.score >= self.config.anomaly_alert_score:
            state = self._state("anomaly")
            state.consecutive += 1
            if state.consecutive >= trigger_n:
                metrics = ", ".join(f"{n} (z={z:.1f})" for n, z in anomaly.tripped[:3])
                detail = f"anomaly score {anomaly.score:.2f}"
                if metrics:
                    detail += f" - unusual: {metrics}"
                report.alerts.append(
                    Alert("anomaly", "warning", "Unusual behaviour", detail, anomaly.score)
                )
                if self._should_announce("anomaly", now):
                    report.events.append(
                        {
                            "kind": "anomaly",
                            "severity": "warning",
                            "key": "anomaly",
                            "message": f"Unusual system behaviour: {detail}",
                            "value": float(anomaly.score),
                        }
                    )
        else:
            self._state("anomaly").consecutive = 0

        # -- forecast-based early warnings -----------------------------
        # Only meaningful when the model is actually trusted; otherwise the
        # "forecast" is just the current value and warns about nothing new.
        for target, threshold in (("cpu", self.config.cpu_high), ("ram", self.config.ram_high)):
            result = forecasts.get(target)
            if result is None or result.predicted is None:
                continue
            if result.trust_weight <= 0.05 or not result.ready:
                continue
            current = sample.cpu if target == "cpu" else sample.ram
            if result.predicted >= threshold > current:
                key = f"forecast_{target}"
                detail = (
                    f"{target.upper()} predicted to reach {result.predicted:.0f}% "
                    f"within {result.horizon}s (now {current:.0f}%)"
                )
                report.alerts.append(Alert(key, "info", "Predicted pressure", detail, result.predicted))
                if self._should_announce(key, now):
                    report.events.append(
                        {
                            "kind": "forecast",
                            "severity": "info",
                            "key": key,
                            "message": detail,
                            "value": float(result.predicted),
                        }
                    )

        # -- exhaustion projections ------------------------------------
        step = self.config.sample_interval
        # ~20 minutes of RAM history: long enough to see a leak, short enough to
        # ignore yesterday's behaviour.
        ram_window = [s.ram for s in history[-1200:]]
        seconds_to_full = project_exhaustion(ram_window, ceiling=95.0, step_seconds=step)
        if seconds_to_full is not None:
            minutes = seconds_to_full / 60.0
            report.ram_exhaustion_minutes = minutes
            if minutes <= 45:
                offender = top_mem[0].name if top_mem else "a background process"
                detail = f"RAM trending up; ~{minutes:.0f} min to 95% at the current rate"
                report.alerts.append(Alert("ram_leak", "warning", "Possible memory leak", detail, minutes))
                report.advice.append(
                    Advice(
                        f"Memory is climbing steadily - {offender} is the largest consumer "
                        f"({top_mem[0].memory_mb:.0f} MB). Restart it if this keeps rising."
                        if top_mem
                        else "Memory is climbing steadily; consider restarting long-running apps.",
                        severity="warning",
                        because=detail,
                    )
                )
                if self._should_announce("ram_leak", now):
                    report.events.append(
                        {
                            "kind": "trend",
                            "severity": "warning",
                            "key": "ram_leak",
                            "message": detail,
                            "value": float(minutes),
                        }
                    )

        disk_window = [s.disk_pct for s in history[-3600:]]
        disk_seconds = project_exhaustion(disk_window, ceiling=100.0, step_seconds=step, min_points=600)
        if disk_seconds is not None:
            hours = disk_seconds / 3600.0
            report.disk_exhaustion_hours = hours
            if hours <= 48:
                report.advice.append(
                    Advice(
                        f"Disk is filling up - about {hours:.0f}h of headroom at the current write rate.",
                        severity="warning",
                        because="linear projection of disk usage over the last hour",
                    )
                )

        # -- actionable advice ------------------------------------------
        firing = {a.key for a in report.alerts}

        if "cpu_high" in firing and top_cpu:
            names = ", ".join(f"{p.name} ({p.cpu_percent:.0f}%)" for p in top_cpu)
            report.advice.append(
                Advice(
                    f"Top CPU consumers right now: {names}. Pause or close whichever you are not using.",
                    severity="warning",
                    because=f"CPU has been above {self.config.cpu_high:.0f}% for several seconds",
                )
            )
        if "ram_high" in firing and top_mem:
            names = ", ".join(f"{p.name} ({p.memory_mb:.0f} MB)" for p in top_mem)
            report.advice.append(
                Advice(
                    f"Largest memory users: {names}. Restarting one of these frees the most RAM.",
                    severity="warning",
                    because=f"RAM above {self.config.ram_high:.0f}%",
                )
            )
        if "swap_high" in firing:
            report.advice.append(
                Advice(
                    "The system is paging memory to disk, which slows everything down far more "
                    "than high CPU. Close something memory-heavy now.",
                    severity="critical",
                    because=f"swap at {sample.swap:.0f}%",
                )
            )
        if "disk_high" in firing:
            report.advice.append(
                Advice(
                    "Free disk space: clear caches, downloads and old build artefacts. "
                    "Keep at least 10% free so the filesystem stays fast.",
                    severity="critical",
                    because=f"disk at {sample.disk_pct:.0f}%",
                )
            )
        if "temp_high" in firing:
            report.advice.append(
                Advice(
                    "Thermals are high, so the CPU may be throttling. Check airflow and vents, "
                    "and reduce sustained load.",
                    severity="warning",
                    because=f"{sample.temp_c:.0f}degC" if sample.temp_c else None,
                )
            )
        if "battery_low" in firing:
            report.advice.append(
                Advice("Plug in soon, or save your work.", severity="warning")
            )

        uptime_days = sample.uptime_seconds / 86400.0
        if uptime_days >= 7:
            report.advice.append(
                Advice(
                    f"Uptime is {uptime_days:.0f} days. A reboot clears leaked memory and "
                    "applies pending updates.",
                    because="long uptime",
                )
            )

        if session_state is not None and session_state.break_due:
            report.advice.append(
                Advice(
                    f"You have been active for {session_state.continuous_minutes:.0f} minutes "
                    "straight - a short break helps.",
                    because="continuous activity",
                )
            )

        if not report.advice and not report.alerts:
            if health >= 85:
                report.advice.append(
                    Advice("Nothing needs attention - resource headroom is healthy.")
                )
            else:
                report.advice.append(
                    Advice("No specific problem detected, but headroom is shrinking. Keep an eye on it.")
                )

        report.alerts.sort(key=lambda a: a.rank)
        report.advice.sort(key=lambda a: a.rank)
        return report
