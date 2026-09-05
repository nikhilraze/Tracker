"""Central configuration.

Every tunable lives here so the sampler, models and UI cannot drift apart.
Values can be overridden with ``NEUROTRACK_*`` environment variables, which is
how the Settings page and the CLI pass overrides without editing code.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path


def _env(name: str) -> str | None:
    return os.environ.get(f"NEUROTRACK_{name.upper()}")


def _env_float(name: str, default: float) -> float:
    raw = _env(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    if raw is None:
        return default
    try:
        return int(float(raw))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = _env(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def default_data_dir() -> Path:
    raw = _env("data_dir")
    if raw:
        return Path(raw).expanduser()
    return Path.home() / ".neurotrack"


@dataclass
class Config:
    """Runtime configuration for the whole application."""

    # ---- sampling -------------------------------------------------------
    # Wall-clock seconds between samples. 1.0 == the per-second requirement.
    sample_interval: float = field(default_factory=lambda: _env_float("sample_interval", 1.0))
    # Per-process scanning is far more expensive than global counters, so it
    # runs on a slower cadence (every N samples) to keep overhead near zero.
    process_scan_every: int = field(default_factory=lambda: _env_int("process_scan_every", 5))
    top_process_count: int = field(default_factory=lambda: _env_int("top_process_count", 8))
    # In-memory ring buffer length. 3600 samples == 1 hour at 1 Hz.
    buffer_size: int = field(default_factory=lambda: _env_int("buffer_size", 3600))

    # ---- storage --------------------------------------------------------
    data_dir: Path = field(default_factory=default_data_dir)
    # Raw 1 Hz rows are bulky (86.4k rows/day), so they expire quickly while
    # 1-minute rollups are kept for long-term trends.
    raw_retention_hours: float = field(default_factory=lambda: _env_float("raw_retention_hours", 12.0))
    rollup_retention_days: float = field(default_factory=lambda: _env_float("rollup_retention_days", 90.0))
    event_retention_days: float = field(default_factory=lambda: _env_float("event_retention_days", 30.0))
    prediction_retention_hours: float = field(
        default_factory=lambda: _env_float("prediction_retention_hours", 24.0)
    )
    # How often the sampler flushes buffered rows to SQLite (in samples).
    flush_every: int = field(default_factory=lambda: _env_int("flush_every", 10))
    maintenance_every: int = field(default_factory=lambda: _env_int("maintenance_every", 300))
    # Seconds between model checkpoints. A pickled IsolationForest is a few
    # hundred KB, so saving it every few seconds would mean gigabytes of
    # pointless disk writes per day on a machine that runs this all the time.
    model_save_interval: float = field(default_factory=lambda: _env_float("model_save_interval", 300.0))

    # ---- forecasting model ---------------------------------------------
    # Predict this many seconds into the future.
    forecast_horizon: int = field(default_factory=lambda: _env_int("forecast_horizon", 15))
    # Minimum labelled pairs before forecasts are presented as ready.
    model_warmup_samples: int = field(default_factory=lambda: _env_int("model_warmup_samples", 120))
    # Rolling-window Ridge settings. All three were chosen empirically in
    # scripts/sweep_ridge.py; per-sample SGD was measured to be worse.
    forecast_window: int = field(default_factory=lambda: _env_int("forecast_window", 2400))
    forecast_refit_every: int = field(default_factory=lambda: _env_int("forecast_refit_every", 30))
    forecast_alpha: float = field(default_factory=lambda: _env_float("forecast_alpha", 100.0))
    # Labelled pairs required before the first fit. At 1 Hz this is ~2 minutes
    # of data on top of the horizon delay.
    forecast_min_fit: int = field(default_factory=lambda: _env_int("forecast_min_fit", 120))
    # Exponential decay factor for the rolling online error metrics.
    metric_decay: float = field(default_factory=lambda: _env_float("metric_decay", 0.995))

    # ---- anomaly model --------------------------------------------------
    anomaly_window: int = field(default_factory=lambda: _env_int("anomaly_window", 1800))
    anomaly_min_train: int = field(default_factory=lambda: _env_int("anomaly_min_train", 180))
    anomaly_retrain_every: int = field(default_factory=lambda: _env_int("anomaly_retrain_every", 300))
    anomaly_contamination: float = field(default_factory=lambda: _env_float("anomaly_contamination", 0.02))
    # Streaming robust z-score above which a single metric is called unusual.
    zscore_threshold: float = field(default_factory=lambda: _env_float("zscore_threshold", 3.5))
    anomaly_alert_score: float = field(default_factory=lambda: _env_float("anomaly_alert_score", 0.75))

    # ---- thresholds / feedback -----------------------------------------
    cpu_high: float = field(default_factory=lambda: _env_float("cpu_high", 85.0))
    cpu_clear: float = field(default_factory=lambda: _env_float("cpu_clear", 65.0))
    ram_high: float = field(default_factory=lambda: _env_float("ram_high", 85.0))
    ram_clear: float = field(default_factory=lambda: _env_float("ram_clear", 70.0))
    disk_high: float = field(default_factory=lambda: _env_float("disk_high", 90.0))
    disk_clear: float = field(default_factory=lambda: _env_float("disk_clear", 85.0))
    swap_high: float = field(default_factory=lambda: _env_float("swap_high", 50.0))
    swap_clear: float = field(default_factory=lambda: _env_float("swap_clear", 25.0))
    temp_high: float = field(default_factory=lambda: _env_float("temp_high", 85.0))
    temp_clear: float = field(default_factory=lambda: _env_float("temp_clear", 75.0))
    battery_low: float = field(default_factory=lambda: _env_float("battery_low", 20.0))
    # An alert must hold for this many consecutive samples before it fires,
    # which suppresses single-sample spikes.
    trigger_samples: int = field(default_factory=lambda: _env_int("trigger_samples", 3))
    # Minimum seconds between repeats of the same alert.
    alert_cooldown: float = field(default_factory=lambda: _env_float("alert_cooldown", 180.0))

    # ---- activity sessions ----------------------------------------------
    # Activity is inferred from resource movement, not keyboard/mouse input.
    activity_cpu_threshold: float = field(default_factory=lambda: _env_float("activity_cpu_threshold", 12.0))
    activity_io_threshold: float = field(default_factory=lambda: _env_float("activity_io_threshold", 0.6))
    # Seconds of calm before an active session is considered finished.
    idle_grace_seconds: float = field(default_factory=lambda: _env_float("idle_grace_seconds", 120.0))
    break_reminder_minutes: float = field(default_factory=lambda: _env_float("break_reminder_minutes", 55.0))
    enable_break_reminders: bool = field(default_factory=lambda: _env_bool("enable_break_reminders", True))

    # ---- health score weights ------------------------------------------
    weight_cpu: float = field(default_factory=lambda: _env_float("weight_cpu", 0.30))
    weight_ram: float = field(default_factory=lambda: _env_float("weight_ram", 0.30))
    weight_disk: float = field(default_factory=lambda: _env_float("weight_disk", 0.15))
    weight_swap: float = field(default_factory=lambda: _env_float("weight_swap", 0.15))
    weight_thermal: float = field(default_factory=lambda: _env_float("weight_thermal", 0.10))

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir).expanduser()
        # Guard against configurations that would busy-spin or break windowing.
        self.sample_interval = max(0.2, float(self.sample_interval))
        self.forecast_horizon = max(1, int(self.forecast_horizon))
        self.buffer_size = max(600, int(self.buffer_size))
        self.process_scan_every = max(1, int(self.process_scan_every))

    # ---- derived paths --------------------------------------------------
    @property
    def db_path(self) -> Path:
        return self.data_dir / "neurotrack.db"

    @property
    def model_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def forecaster_path(self) -> Path:
        return self.model_dir / "forecaster.joblib"

    @property
    def anomaly_path(self) -> Path:
        return self.model_dir / "anomaly.joblib"

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def as_dict(self) -> dict:
        data = asdict(self)
        data["data_dir"] = str(self.data_dir)
        return data


def load_config() -> Config:
    """Build a config from environment overrides and make sure dirs exist."""
    cfg = Config()
    cfg.ensure_dirs()
    return cfg
