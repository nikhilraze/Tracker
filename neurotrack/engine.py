"""The monitoring engine.

This is the piece that makes per-second tracking actually work. The original
app sampled inside the Streamlit script with ``time.sleep(5); st.rerun()``, which
meant:

* nothing was recorded unless a browser tab was open;
* a row was written on *every* rerun, so clicking a page logged a duplicate;
* the whole UI blocked for the duration of the sleep.

Here a single daemon thread owns the sampling loop at a fixed cadence and the UI
is a pure reader of the latest snapshot. Sampling continues while the browser is
closed, and no UI interaction can corrupt the data.

Per tick the thread does: collect -> build features -> grade the due prediction
and retrain -> forecast forward -> score anomaly -> evaluate feedback -> buffer
for storage. That whole chain measures a few milliseconds.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from .collector import (
    MetricCollector,
    ProcessInfo,
    Sample,
    sample_from_row,
    static_system_info,
)
from .config import Config, load_config
from .features import build_anomaly_features, build_forecast_features
from .feedback import FeedbackEngine, FeedbackReport
from .models import AnomalyModel, AnomalyResult, ForecastService, health_score
from .sessions import SessionState, SessionTracker
from .storage import Storage


@dataclass
class Snapshot:
    """Everything the UI needs for one render, captured atomically."""

    sample: Sample
    health: int
    anomaly: AnomalyResult | None
    forecasts: dict = field(default_factory=dict)
    processes: list[ProcessInfo] = field(default_factory=list)
    feedback: FeedbackReport | None = None
    session: SessionState | None = None
    tick: int = 0
    sampled_at: float = 0.0


class MonitorEngine:
    """Owns the sampler thread, the models, and the shared latest snapshot."""

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or load_config()
        self.config.ensure_dirs()

        self.storage = Storage(self.config.db_path)
        self.collector = MetricCollector(top_n=self.config.top_process_count)
        self.system_info = static_system_info()

        self.forecast = ForecastService(
            horizon=self.config.forecast_horizon,
            decay=self.config.metric_decay,
            warmup=self.config.model_warmup_samples,
            sample_interval=self.config.sample_interval,
            window=self.config.forecast_window,
            refit_every=self.config.forecast_refit_every,
            alpha=self.config.forecast_alpha,
            min_fit=self.config.forecast_min_fit,
        )
        self.anomaly = AnomalyModel(
            window=self.config.anomaly_window,
            min_train=self.config.anomaly_min_train,
            retrain_every=self.config.anomaly_retrain_every,
            contamination=self.config.anomaly_contamination,
            z_threshold=self.config.zscore_threshold,
        )
        self.feedback = FeedbackEngine(self.config)
        self.sessions = SessionTracker(
            self.storage,
            cpu_threshold=self.config.activity_cpu_threshold,
            io_threshold=self.config.activity_io_threshold,
            idle_grace_seconds=self.config.idle_grace_seconds,
            break_reminder_minutes=self.config.break_reminder_minutes,
            sample_interval=self.config.sample_interval,
        )

        self.models_restored = {
            "forecaster": self.forecast.load(self.config.forecaster_path),
            "anomaly": self.anomaly.load(self.config.anomaly_path),
        }

        self._history: deque[Sample] = deque(maxlen=self.config.buffer_size)
        self._pending_rows: list[dict] = []
        self._processes: list[ProcessInfo] = []
        self._snapshot: Snapshot | None = None

        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

        self.tick = 0
        self.started_at: float | None = None
        self.last_error: str | None = None
        self.error_count = 0
        self.skipped_ticks = 0
        self._last_model_save = 0.0
        self._saved_anomaly_fit = -1

        self._warm_start()

    # ------------------------------------------------------------------
    # startup
    # ------------------------------------------------------------------
    def _warm_start(self) -> None:
        """Reload recent history so charts and models are not empty on boot."""
        try:
            rows = self.storage.recent_samples(limit=self.config.buffer_size)
        except Exception:
            rows = []
        ram_total = self.system_info.get("ram_total_gb", 0.0)
        for row in rows:
            try:
                self._history.append(sample_from_row(row, ram_total))
            except Exception:
                continue

        # Seed the anomaly window from restored history so it can fit at once.
        if not self.models_restored["anomaly"] and len(self._history) >= 60:
            history = list(self._history)
            vectors = [
                build_anomaly_features(history, i)
                for i in range(len(history))
            ]
            try:
                self.anomaly.bulk_warm(vectors)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            self.started_at = time.time()
            self._thread = threading.Thread(
                target=self._run, name="neurotrack-sampler", daemon=True
            )
            self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
        self._flush(force=True)
        self.sessions.shutdown()
        # Always checkpoint on a clean shutdown so learning is not lost.
        self.save_models(force=True)

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------
    # main loop
    # ------------------------------------------------------------------
    def _run(self) -> None:
        interval = self.config.sample_interval
        # Absolute schedule rather than `sleep(interval)`, so processing time
        # does not make the cadence drift.
        next_tick = time.monotonic()
        while not self._stop.is_set():
            next_tick += interval
            try:
                self._tick()
            except Exception as exc:  # never let one bad tick kill the thread
                self.error_count += 1
                self.last_error = f"{type(exc).__name__}: {exc}"

            delay = next_tick - time.monotonic()
            if delay < 0:
                # Fell behind (machine suspended, or a very slow tick). Skip
                # ahead instead of firing a burst of catch-up samples.
                missed = int(-delay // interval) + 1
                self.skipped_ticks += missed
                next_tick += missed * interval
                delay = max(0.0, next_tick - time.monotonic())
            self._stop.wait(delay)

    def _tick(self) -> None:
        sample = self.collector.sample()
        self.tick += 1

        with self._lock:
            self._history.append(sample)
            history = list(self._history)

        idx = len(history) - 1

        # 1. features
        features = build_forecast_features(history, idx)

        # 2. grade the prediction whose label just arrived, retrain, re-forecast
        forecasts = self.forecast.observe(
            features, {"cpu": sample.cpu, "ram": sample.ram}, sample.ts
        )

        # 3. anomaly scoring
        anomaly_vector = build_anomaly_features(history, idx)
        anomaly = self.anomaly.observe(
            anomaly_vector,
            {
                "cpu": sample.cpu,
                "ram": sample.ram,
                "swap": sample.swap,
                "disk_io_mbps": sample.disk_io_mbps,
                "net_io_mbps": sample.net_io_mbps,
            },
        )

        # 4. per-process attribution on a slower cadence (it is the expensive part)
        if self.tick % self.config.process_scan_every == 0 or not self._processes:
            try:
                self._processes = self.collector.top_processes()
            except Exception:
                pass

        # 5. sessions
        session_state, session_events = self.sessions.observe(sample)

        # 6. health + feedback
        health = health_score(
            sample.cpu,
            sample.ram,
            sample.disk_pct,
            sample.swap,
            sample.temp_c,
            weights={
                "cpu": self.config.weight_cpu,
                "ram": self.config.weight_ram,
                "disk": self.config.weight_disk,
                "swap": self.config.weight_swap,
                "thermal": self.config.weight_thermal,
            },
        )
        report = self.feedback.evaluate(
            sample, history, forecasts, anomaly, self._processes, session_state, health
        )

        # 7. publish the snapshot the UI reads
        snapshot = Snapshot(
            sample=sample,
            health=health,
            anomaly=anomaly,
            forecasts=forecasts,
            processes=list(self._processes),
            feedback=report,
            session=session_state,
            tick=self.tick,
            sampled_at=sample.ts,
        )
        with self._lock:
            self._snapshot = snapshot

        # 8. persist
        row = sample.as_dict()
        row["anomaly_score"] = anomaly.score if anomaly else None
        row["health_score"] = health
        row["active"] = 1 if session_state.active else 0
        self._pending_rows.append(row)

        for event in list(report.events) + list(session_events):
            try:
                self.storage.add_event(
                    kind=event["kind"],
                    severity=event["severity"],
                    message=event["message"],
                    key=event.get("key"),
                    value=event.get("value"),
                    ts=sample.ts,
                )
            except Exception:
                pass

        graded = self.forecast.drain_graded_rows()
        if graded:
            try:
                self.storage.record_predictions(graded)
            except Exception:
                pass

        self._flush()

        if self.tick % self.config.maintenance_every == 0:
            self._maintain()

    # ------------------------------------------------------------------
    # persistence helpers
    # ------------------------------------------------------------------
    def _flush(self, force: bool = False) -> None:
        if not self._pending_rows:
            return
        if not force and len(self._pending_rows) < self.config.flush_every:
            return
        rows, self._pending_rows = self._pending_rows, []
        try:
            self.storage.insert_samples(rows)
        except Exception as exc:
            self.error_count += 1
            self.last_error = f"storage: {exc}"

    def _maintain(self) -> None:
        """Rollups, retention and model checkpointing."""
        try:
            self.storage.build_rollups()
            self.storage.prune(
                raw_retention_hours=self.config.raw_retention_hours,
                rollup_retention_days=self.config.rollup_retention_days,
                event_retention_days=self.config.event_retention_days,
                prediction_retention_hours=self.config.prediction_retention_hours,
            )
        except Exception as exc:
            self.error_count += 1
            self.last_error = f"maintenance: {exc}"
        self.save_models()

    def save_models(self, force: bool = False) -> None:
        """Checkpoint models, rate-limited and skipping unchanged ones."""
        now = time.time()
        if not force and now - self._last_model_save < self.config.model_save_interval:
            return
        self._last_model_save = now
        try:
            self.forecast.save(self.config.forecaster_path)
            # The forest is the expensive object to serialise, so only rewrite it
            # when it has actually been refit since the last checkpoint.
            if force or self.anomaly.train_count != self._saved_anomaly_fit:
                self.anomaly.save(self.config.anomaly_path)
                self._saved_anomaly_fit = self.anomaly.train_count
        except Exception as exc:
            self.error_count += 1
            self.last_error = f"model save: {exc}"

    # ------------------------------------------------------------------
    # read API for the UI
    # ------------------------------------------------------------------
    def snapshot(self) -> Snapshot | None:
        with self._lock:
            return self._snapshot

    def history(self, limit: int | None = None) -> list[Sample]:
        with self._lock:
            data = list(self._history)
        return data[-limit:] if limit else data

    def status(self) -> dict[str, Any]:
        snapshot = self.snapshot()
        age = None if snapshot is None else max(0.0, time.time() - snapshot.sampled_at)
        return {
            "running": self.running,
            "tick": self.tick,
            "started_at": self.started_at,
            "uptime_seconds": None if self.started_at is None else time.time() - self.started_at,
            "sample_interval": self.config.sample_interval,
            "buffer": len(self._history),
            "snapshot_age": age,
            "skipped_ticks": self.skipped_ticks,
            "errors": self.error_count,
            "last_error": self.last_error,
            "models_restored": self.models_restored,
            "db_mb": self.storage.database_size_mb(),
            "stale_labels_dropped": self.forecast.dropped_stale,
        }


# ---------------------------------------------------------------------------
# process-wide singleton
# ---------------------------------------------------------------------------
_engine_lock = threading.Lock()
_engine: MonitorEngine | None = None


def get_engine(config: Config | None = None) -> MonitorEngine:
    """Return the shared engine, starting the sampler on first use.

    Streamlit re-executes the script on every interaction and for every browser
    session, so the engine must be created once per *process*, not per run. The
    UI layer wraps this in ``st.cache_resource``; the lock here protects against
    two sessions racing on first load.
    """
    global _engine
    with _engine_lock:
        if _engine is None:
            _engine = MonitorEngine(config)
            _engine.start()
        return _engine
