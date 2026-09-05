"""SQLite persistence.

The old CSV-append approach cannot survive per-second sampling: 1 Hz produces
86,400 rows per day, and every read re-parsed the entire file. This module
stores raw samples in SQLite and continuously folds them into 1-minute rollups,
so long-range charts read a few hundred rows instead of millions.

Retention is tiered:
  * ``samples``     - raw 1 Hz rows, kept for hours (high volume, high detail)
  * ``samples_1m``  - per-minute aggregates, kept for months (low volume)

A single connection is shared behind a lock. Writes only ever come from the
sampler thread; Streamlit reruns are readers.
"""

from __future__ import annotations

import json
import math
import sqlite3
import threading
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

# Columns of the raw `samples` table, in insertion order.
SAMPLE_COLUMNS: tuple[str, ...] = (
    "ts",
    "cpu",
    "cpu_max_core",
    "ram",
    "ram_used_gb",
    "swap",
    "disk_pct",
    "disk_read_mbps",
    "disk_write_mbps",
    "net_sent_mbps",
    "net_recv_mbps",
    "load1_per_core",
    "proc_count",
    "cpu_freq_mhz",
    "battery_pct",
    "battery_plugged",
    "temp_c",
    "fan_rpm",
    "uptime_seconds",
    "anomaly_score",
    "health_score",
    "active",
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS samples (
    ts REAL PRIMARY KEY,
    cpu REAL, cpu_max_core REAL,
    ram REAL, ram_used_gb REAL, swap REAL,
    disk_pct REAL, disk_read_mbps REAL, disk_write_mbps REAL,
    net_sent_mbps REAL, net_recv_mbps REAL,
    load1_per_core REAL, proc_count INTEGER, cpu_freq_mhz REAL,
    battery_pct REAL, battery_plugged INTEGER,
    temp_c REAL, fan_rpm INTEGER,
    uptime_seconds REAL,
    anomaly_score REAL, health_score REAL, active INTEGER
);

CREATE TABLE IF NOT EXISTS samples_1m (
    bucket_ts INTEGER PRIMARY KEY,
    n INTEGER,
    cpu_avg REAL, cpu_max REAL,
    ram_avg REAL, ram_max REAL,
    swap_avg REAL, disk_pct_avg REAL,
    disk_io_mbps_avg REAL, net_io_mbps_avg REAL,
    temp_avg REAL, temp_max REAL,
    battery_avg REAL,
    health_avg REAL, anomaly_max REAL,
    active_seconds REAL
);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    kind TEXT NOT NULL,
    severity TEXT NOT NULL,
    key TEXT,
    message TEXT NOT NULL,
    value REAL,
    detail TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);

CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    start_ts REAL NOT NULL,
    end_ts REAL,
    active_seconds REAL DEFAULT 0,
    cpu_avg REAL, ram_avg REAL,
    samples INTEGER DEFAULT 0,
    closed INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_sessions_start ON sessions(start_ts);

CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    target TEXT NOT NULL,
    horizon INTEGER NOT NULL,
    predicted REAL,
    actual REAL,
    abs_err REAL,
    naive_abs_err REAL
);
CREATE INDEX IF NOT EXISTS idx_predictions_ts ON predictions(ts);

CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT
);
"""


def _clean(value: Any) -> Any:
    """SQLite rejects NaN/inf silently-ish; normalise to NULL."""
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


class Storage:
    """Thread-safe SQLite wrapper for telemetry, events and model bookkeeping."""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            str(self.db_path), check_same_thread=False, timeout=30.0
        )
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            # WAL lets UI readers work while the sampler writes.
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.commit()
            finally:
                self._conn.close()

    # ------------------------------------------------------------------
    # samples
    # ------------------------------------------------------------------
    def insert_samples(self, rows: Iterable[dict]) -> int:
        """Insert raw samples. Duplicate timestamps are ignored, not fatal."""
        payload = [tuple(_clean(row.get(col)) for col in SAMPLE_COLUMNS) for row in rows]
        if not payload:
            return 0
        placeholders = ",".join("?" * len(SAMPLE_COLUMNS))
        sql = (
            f"INSERT OR IGNORE INTO samples ({','.join(SAMPLE_COLUMNS)}) "
            f"VALUES ({placeholders})"
        )
        with self._lock:
            cur = self._conn.executemany(sql, payload)
            self._conn.commit()
            return cur.rowcount or 0

    def recent_samples(self, limit: int = 600) -> list[sqlite3.Row]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM samples ORDER BY ts DESC LIMIT ?", (int(limit),)
            )
            return list(reversed(cur.fetchall()))

    def samples_since(self, since_ts: float) -> list[sqlite3.Row]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM samples WHERE ts >= ? ORDER BY ts ASC", (float(since_ts),)
            )
            return cur.fetchall()

    def sample_count(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0])

    def total_sampled_seconds(self) -> float:
        """Sum of observed time across rollups (survives raw-row pruning)."""
        with self._lock:
            row = self._conn.execute("SELECT COALESCE(SUM(n),0) FROM samples_1m").fetchone()
            return float(row[0] or 0.0)

    # ------------------------------------------------------------------
    # rollups
    # ------------------------------------------------------------------
    def build_rollups(self, now: float | None = None) -> int:
        """Aggregate completed minutes from `samples` into `samples_1m`.

        Only minute buckets that have fully elapsed are written, so a bucket is
        never persisted while still accumulating samples.
        """
        now = now or time.time()
        newest_complete = (int(now) // 60) * 60  # start of the current minute
        with self._lock:
            cur = self._conn.execute(
                """
                INSERT OR REPLACE INTO samples_1m (
                    bucket_ts, n, cpu_avg, cpu_max, ram_avg, ram_max,
                    swap_avg, disk_pct_avg, disk_io_mbps_avg, net_io_mbps_avg,
                    temp_avg, temp_max, battery_avg, health_avg, anomaly_max,
                    active_seconds
                )
                SELECT
                    CAST(ts / 60 AS INTEGER) * 60 AS bucket,
                    COUNT(*),
                    AVG(cpu), MAX(cpu),
                    AVG(ram), MAX(ram),
                    AVG(swap), AVG(disk_pct),
                    AVG(COALESCE(disk_read_mbps,0) + COALESCE(disk_write_mbps,0)),
                    AVG(COALESCE(net_sent_mbps,0) + COALESCE(net_recv_mbps,0)),
                    AVG(temp_c), MAX(temp_c), AVG(battery_pct),
                    AVG(health_score), MAX(anomaly_score),
                    SUM(COALESCE(active,0))
                FROM samples
                WHERE ts < ?
                GROUP BY bucket
                """,
                (float(newest_complete),),
            )
            self._conn.commit()
            return cur.rowcount or 0

    def rollups_since(self, since_ts: float) -> list[sqlite3.Row]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM samples_1m WHERE bucket_ts >= ? ORDER BY bucket_ts ASC",
                (float(since_ts),),
            )
            return cur.fetchall()

    def daily_activity(self, days: int = 14) -> list[sqlite3.Row]:
        """Active seconds and average load per local calendar day."""
        cutoff = time.time() - days * 86400
        with self._lock:
            cur = self._conn.execute(
                """
                SELECT date(bucket_ts, 'unixepoch', 'localtime') AS day,
                       SUM(active_seconds) AS active_seconds,
                       SUM(n) AS observed_seconds,
                       AVG(cpu_avg) AS cpu_avg,
                       AVG(ram_avg) AS ram_avg,
                       MAX(cpu_max) AS cpu_max
                FROM samples_1m
                WHERE bucket_ts >= ?
                GROUP BY day
                ORDER BY day ASC
                """,
                (float(cutoff),),
            )
            return cur.fetchall()

    def hourly_profile(self, days: int = 14) -> list[sqlite3.Row]:
        """Average load by hour-of-day - the basis of 'your usual rhythm'."""
        cutoff = time.time() - days * 86400
        with self._lock:
            cur = self._conn.execute(
                """
                SELECT CAST(strftime('%H', bucket_ts, 'unixepoch', 'localtime') AS INTEGER) AS hour,
                       AVG(cpu_avg) AS cpu_avg,
                       AVG(ram_avg) AS ram_avg,
                       SUM(active_seconds) AS active_seconds,
                       SUM(n) AS observed_seconds
                FROM samples_1m
                WHERE bucket_ts >= ?
                GROUP BY hour
                ORDER BY hour ASC
                """,
                (float(cutoff),),
            )
            return cur.fetchall()

    # ------------------------------------------------------------------
    # events
    # ------------------------------------------------------------------
    def add_event(
        self,
        kind: str,
        severity: str,
        message: str,
        key: str | None = None,
        value: float | None = None,
        detail: dict | None = None,
        ts: float | None = None,
    ) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO events (ts, kind, severity, key, message, value, detail)"
                " VALUES (?,?,?,?,?,?,?)",
                (
                    float(ts or time.time()),
                    kind,
                    severity,
                    key,
                    message,
                    _clean(value),
                    json.dumps(detail) if detail else None,
                ),
            )
            self._conn.commit()

    def recent_events(self, limit: int = 50, since_ts: float | None = None) -> list[sqlite3.Row]:
        with self._lock:
            if since_ts is None:
                cur = self._conn.execute(
                    "SELECT * FROM events ORDER BY ts DESC LIMIT ?", (int(limit),)
                )
            else:
                cur = self._conn.execute(
                    "SELECT * FROM events WHERE ts >= ? ORDER BY ts DESC LIMIT ?",
                    (float(since_ts), int(limit)),
                )
            return cur.fetchall()

    def event_summary(self, since_ts: float) -> list[sqlite3.Row]:
        with self._lock:
            cur = self._conn.execute(
                """
                SELECT key, kind, severity, COUNT(*) AS hits, MAX(ts) AS last_ts
                FROM events WHERE ts >= ? AND key IS NOT NULL
                GROUP BY key, kind, severity
                ORDER BY hits DESC
                """,
                (float(since_ts),),
            )
            return cur.fetchall()

    # ------------------------------------------------------------------
    # sessions
    # ------------------------------------------------------------------
    def open_session(self, start_ts: float) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO sessions (start_ts, end_ts, closed) VALUES (?,?,0)",
                (float(start_ts), float(start_ts)),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def update_session(
        self,
        session_id: int,
        end_ts: float,
        active_seconds: float,
        cpu_avg: float,
        ram_avg: float,
        samples: int,
        closed: bool = False,
    ) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE sessions SET end_ts=?, active_seconds=?, cpu_avg=?, ram_avg=?,"
                " samples=?, closed=? WHERE id=?",
                (
                    float(end_ts),
                    float(active_seconds),
                    _clean(float(cpu_avg)),
                    _clean(float(ram_avg)),
                    int(samples),
                    int(closed),
                    int(session_id),
                ),
            )
            self._conn.commit()

    def recent_sessions(self, limit: int = 20) -> list[sqlite3.Row]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM sessions ORDER BY start_ts DESC LIMIT ?", (int(limit),)
            )
            return cur.fetchall()

    def close_dangling_sessions(self) -> None:
        """Close sessions left open by a crash or hard shutdown."""
        with self._lock:
            self._conn.execute("UPDATE sessions SET closed=1 WHERE closed=0")
            self._conn.commit()

    # ------------------------------------------------------------------
    # predictions (model audit trail)
    # ------------------------------------------------------------------
    def record_predictions(self, rows: Sequence[tuple]) -> None:
        """rows: (ts, target, horizon, predicted, actual, abs_err, naive_abs_err)"""
        if not rows:
            return
        with self._lock:
            self._conn.executemany(
                "INSERT INTO predictions (ts, target, horizon, predicted, actual,"
                " abs_err, naive_abs_err) VALUES (?,?,?,?,?,?,?)",
                [tuple(_clean(v) for v in row) for row in rows],
            )
            self._conn.commit()

    def prediction_scores(self, target: str, limit: int = 500) -> list[sqlite3.Row]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM predictions WHERE target=? ORDER BY ts DESC LIMIT ?",
                (target, int(limit)),
            )
            return list(reversed(cur.fetchall()))

    def prediction_summary(self, since_ts: float) -> list[sqlite3.Row]:
        """Model vs naive baseline error, which is the real quality signal."""
        with self._lock:
            cur = self._conn.execute(
                """
                SELECT target, COUNT(*) AS n,
                       AVG(abs_err) AS mae,
                       AVG(naive_abs_err) AS naive_mae
                FROM predictions WHERE ts >= ?
                GROUP BY target
                """,
                (float(since_ts),),
            )
            return cur.fetchall()

    # ------------------------------------------------------------------
    # meta
    # ------------------------------------------------------------------
    def set_meta(self, key: str, value: Any) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO meta (key, value) VALUES (?,?)"
                " ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, json.dumps(value)),
            )
            self._conn.commit()

    def get_meta(self, key: str, default: Any = None) -> Any:
        with self._lock:
            row = self._conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        if row is None:
            return default
        try:
            return json.loads(row[0])
        except (TypeError, ValueError):
            return default

    # ------------------------------------------------------------------
    # maintenance
    # ------------------------------------------------------------------
    def prune(
        self,
        raw_retention_hours: float,
        rollup_retention_days: float,
        event_retention_days: float,
        prediction_retention_hours: float,
        now: float | None = None,
    ) -> dict[str, int]:
        """Delete expired rows. Raw samples are only dropped once rolled up."""
        now = now or time.time()
        deleted: dict[str, int] = {}
        with self._lock:
            row = self._conn.execute("SELECT MAX(bucket_ts) FROM samples_1m").fetchone()
            newest_rollup = row[0]
            raw_cutoff = now - raw_retention_hours * 3600
            if newest_rollup is not None:
                # Never delete raw rows that have not been aggregated yet.
                raw_cutoff = min(raw_cutoff, float(newest_rollup))
            cur = self._conn.execute("DELETE FROM samples WHERE ts < ?", (float(raw_cutoff),))
            deleted["samples"] = cur.rowcount or 0

            cur = self._conn.execute(
                "DELETE FROM samples_1m WHERE bucket_ts < ?",
                (float(now - rollup_retention_days * 86400),),
            )
            deleted["samples_1m"] = cur.rowcount or 0

            cur = self._conn.execute(
                "DELETE FROM events WHERE ts < ?",
                (float(now - event_retention_days * 86400),),
            )
            deleted["events"] = cur.rowcount or 0

            cur = self._conn.execute(
                "DELETE FROM predictions WHERE ts < ?",
                (float(now - prediction_retention_hours * 3600),),
            )
            deleted["predictions"] = cur.rowcount or 0
            self._conn.commit()
        return deleted

    def vacuum(self) -> None:
        with self._lock:
            self._conn.execute("VACUUM")
            self._conn.commit()

    def database_size_mb(self) -> float:
        total = 0.0
        for suffix in ("", "-wal", "-shm"):
            path = Path(str(self.db_path) + suffix)
            if path.exists():
                total += path.stat().st_size
        return total / (1024 * 1024)
