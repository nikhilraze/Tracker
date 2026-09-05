"""Metric collection.

One :class:`MetricCollector` instance is owned by the sampler thread. It is
stateful on purpose: throughput figures (disk / network MB per second) are
derived from the *difference* between successive counter readings, and
``psutil`` percentage helpers are relative to the previous call, so the object
must be reused rather than recreated per sample.

Everything degrades gracefully: hardware with no battery, no fan or no thermal
sensor simply yields ``None`` instead of raising.
"""

from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass

import psutil

# Sensor groups that usually carry the real CPU package temperature, in
# descending order of preference.
_PREFERRED_TEMP_KEYS = (
    "coretemp",
    "k10temp",
    "zenpower",
    "cpu_thermal",
    "cpu-thermal",
    "soc_thermal",
    "acpitz",
)

_MB = 1024.0 * 1024.0


@dataclass
class Sample:
    """A single point-in-time reading of the machine."""

    ts: float
    cpu: float
    cpu_max_core: float
    ram: float
    ram_used_gb: float
    ram_total_gb: float
    swap: float
    disk_pct: float
    disk_read_mbps: float
    disk_write_mbps: float
    net_sent_mbps: float
    net_recv_mbps: float
    load1_per_core: float
    proc_count: int
    cpu_freq_mhz: float | None = None
    battery_pct: float | None = None
    battery_plugged: bool | None = None
    battery_secs_left: int | None = None
    temp_c: float | None = None
    fan_rpm: int | None = None
    uptime_seconds: float = 0.0

    @property
    def disk_io_mbps(self) -> float:
        return self.disk_read_mbps + self.disk_write_mbps

    @property
    def net_io_mbps(self) -> float:
        return self.net_sent_mbps + self.net_recv_mbps

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class ProcessInfo:
    """Resource usage of one process, used to make advice actionable."""

    pid: int
    name: str
    cpu_percent: float
    cpu_percent_normalized: float
    memory_mb: float
    memory_percent: float

    def as_dict(self) -> dict:
        return asdict(self)


def _root_path() -> str:
    """Filesystem root that works on POSIX ('/') and Windows ('C:\\')."""
    return os.path.abspath(os.sep)


def _pick_temperature() -> tuple[float | None, str | None]:
    """Best-effort CPU temperature in Celsius plus the sensor label used."""
    getter = getattr(psutil, "sensors_temperatures", None)
    if getter is None:
        return None, None
    try:
        readings = getter()
    except Exception:
        return None, None
    if not readings:
        return None, None

    def best_of(entries) -> float | None:
        values = [e.current for e in entries if getattr(e, "current", None) is not None and e.current > 0]
        return max(values) if values else None

    for key in _PREFERRED_TEMP_KEYS:
        for name, entries in readings.items():
            if name.lower().startswith(key):
                value = best_of(entries)
                if value is not None:
                    return float(value), name
    # No known CPU sensor: fall back to the hottest sensor of any kind.
    hottest: float | None = None
    label: str | None = None
    for name, entries in readings.items():
        value = best_of(entries)
        if value is not None and (hottest is None or value > hottest):
            hottest, label = value, name
    return (float(hottest), label) if hottest is not None else (None, None)


def _pick_fan_rpm() -> int | None:
    getter = getattr(psutil, "sensors_fans", None)
    if getter is None:
        return None
    try:
        readings = getter()
    except Exception:
        return None
    speeds = [
        entry.current
        for entries in (readings or {}).values()
        for entry in entries
        if getattr(entry, "current", None)
    ]
    return int(max(speeds)) if speeds else None


class MetricCollector:
    """Stateful sampler for global system metrics and per-process usage."""

    def __init__(self, top_n: int = 8) -> None:
        self.top_n = top_n
        self._root = _root_path()
        self._cpu_count = psutil.cpu_count(logical=True) or 1
        self._proc_cache: dict[int, psutil.Process] = {}
        self._last_disk = None
        self._last_net = None
        self._last_ts: float | None = None
        self.temp_sensor_label: str | None = None
        self._prime()

    def _prime(self) -> None:
        """Take a throwaway reading so the first real sample is meaningful.

        ``cpu_percent(interval=None)`` reports usage since the previous call, so
        without priming the very first sample would always read 0%.
        """
        try:
            psutil.cpu_percent(interval=None)
            psutil.cpu_percent(interval=None, percpu=True)
        except Exception:
            pass
        self._last_disk = self._safe_disk_counters()
        self._last_net = self._safe_net_counters()
        self._last_ts = time.time()

    @staticmethod
    def _safe_disk_counters():
        try:
            return psutil.disk_io_counters()
        except Exception:
            return None

    @staticmethod
    def _safe_net_counters():
        try:
            return psutil.net_io_counters()
        except Exception:
            return None

    def _rates(self, now: float) -> tuple[float, float, float, float]:
        """Disk read/write and network sent/recv in MB/s since the last call."""
        elapsed = max(1e-6, now - (self._last_ts or now))

        disk = self._safe_disk_counters()
        read_mbps = write_mbps = 0.0
        if disk is not None and self._last_disk is not None:
            read_delta = disk.read_bytes - self._last_disk.read_bytes
            write_delta = disk.write_bytes - self._last_disk.write_bytes
            # Counters reset when a device disappears or wraps; clamp negatives.
            read_mbps = max(0.0, read_delta) / _MB / elapsed
            write_mbps = max(0.0, write_delta) / _MB / elapsed
        if disk is not None:
            self._last_disk = disk

        net = self._safe_net_counters()
        sent_mbps = recv_mbps = 0.0
        if net is not None and self._last_net is not None:
            sent_delta = net.bytes_sent - self._last_net.bytes_sent
            recv_delta = net.bytes_recv - self._last_net.bytes_recv
            sent_mbps = max(0.0, sent_delta) / _MB / elapsed
            recv_mbps = max(0.0, recv_delta) / _MB / elapsed
        if net is not None:
            self._last_net = net

        self._last_ts = now
        return read_mbps, write_mbps, sent_mbps, recv_mbps

    def sample(self) -> Sample:
        """Collect one :class:`Sample`. Never raises for missing hardware."""
        now = time.time()
        read_mbps, write_mbps, sent_mbps, recv_mbps = self._rates(now)

        cpu = float(psutil.cpu_percent(interval=None))
        try:
            per_core = psutil.cpu_percent(interval=None, percpu=True)
            cpu_max_core = float(max(per_core)) if per_core else cpu
        except Exception:
            cpu_max_core = cpu

        mem = psutil.virtual_memory()
        try:
            swap = float(psutil.swap_memory().percent)
        except Exception:
            swap = 0.0

        try:
            disk_pct = float(psutil.disk_usage(self._root).percent)
        except Exception:
            disk_pct = 0.0

        try:
            load1 = os.getloadavg()[0] / self._cpu_count * 100.0
        except (OSError, AttributeError):
            # Windows has no load average; CPU% is the closest analogue.
            load1 = cpu

        try:
            proc_count = len(psutil.pids())
        except Exception:
            proc_count = 0

        freq_mhz: float | None = None
        try:
            freq = psutil.cpu_freq()
            if freq is not None and freq.current:
                freq_mhz = float(freq.current)
        except Exception:
            freq_mhz = None

        battery_pct = battery_plugged = battery_secs = None
        getter = getattr(psutil, "sensors_battery", None)
        if getter is not None:
            try:
                battery = getter()
                if battery is not None:
                    battery_pct = float(battery.percent)
                    battery_plugged = bool(battery.power_plugged)
                    secs = battery.secsleft
                    if isinstance(secs, int) and secs >= 0:
                        battery_secs = int(secs)
            except Exception:
                pass

        temp_c, label = _pick_temperature()
        if label:
            self.temp_sensor_label = label

        try:
            uptime = max(0.0, now - psutil.boot_time())
        except Exception:
            uptime = 0.0

        return Sample(
            ts=now,
            cpu=cpu,
            cpu_max_core=cpu_max_core,
            ram=float(mem.percent),
            ram_used_gb=(mem.total - mem.available) / (1024**3),
            ram_total_gb=mem.total / (1024**3),
            swap=swap,
            disk_pct=disk_pct,
            disk_read_mbps=read_mbps,
            disk_write_mbps=write_mbps,
            net_sent_mbps=sent_mbps,
            net_recv_mbps=recv_mbps,
            load1_per_core=float(load1),
            proc_count=proc_count,
            cpu_freq_mhz=freq_mhz,
            battery_pct=battery_pct,
            battery_plugged=battery_plugged,
            battery_secs_left=battery_secs,
            temp_c=temp_c,
            fan_rpm=_pick_fan_rpm(),
            uptime_seconds=uptime,
        )

    def top_processes(self) -> list[ProcessInfo]:
        """Heaviest processes right now, ranked by CPU then memory.

        ``Process`` objects are cached between scans because per-process
        ``cpu_percent()`` is also measured relative to the previous call on that
        same object - rebuilding the list each time would report 0% forever.
        """
        seen: set[int] = set()
        rows: list[ProcessInfo] = []

        try:
            pids = psutil.pids()
        except Exception:
            return []

        for pid in pids:
            seen.add(pid)
            proc = self._proc_cache.get(pid)
            if proc is None:
                try:
                    proc = psutil.Process(pid)
                    self._proc_cache[pid] = proc
                    # Prime this process and skip it for one round.
                    proc.cpu_percent(None)
                    continue
                except Exception:
                    self._proc_cache.pop(pid, None)
                    continue
            try:
                with proc.oneshot():
                    cpu_pct = float(proc.cpu_percent(None))
                    mem_info = proc.memory_info()
                    mem_pct = float(proc.memory_percent())
                    name = proc.name()
                rows.append(
                    ProcessInfo(
                        pid=pid,
                        name=name or f"pid-{pid}",
                        cpu_percent=cpu_pct,
                        cpu_percent_normalized=cpu_pct / self._cpu_count,
                        memory_mb=mem_info.rss / _MB,
                        memory_percent=mem_pct,
                    )
                )
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                self._proc_cache.pop(pid, None)
            except Exception:
                continue

        # Drop cache entries for processes that have exited.
        for pid in list(self._proc_cache):
            if pid not in seen:
                self._proc_cache.pop(pid, None)

        rows.sort(key=lambda r: (r.cpu_percent, r.memory_mb), reverse=True)
        return rows[: self.top_n]

    @property
    def cpu_count(self) -> int:
        return self._cpu_count


def sample_from_row(row, ram_total_gb: float = 0.0) -> Sample:
    """Rebuild a :class:`Sample` from a stored ``samples`` row.

    Shared by the engine's warm start and the offline trainer so that replayed
    history is byte-identical to what was originally collected.
    """

    def get(key, default=None):
        try:
            value = row[key]
        except (KeyError, IndexError, TypeError):
            return default
        return default if value is None else value

    plugged = get("battery_plugged")
    return Sample(
        ts=float(get("ts", 0.0)),
        cpu=float(get("cpu", 0.0)),
        cpu_max_core=float(get("cpu_max_core", 0.0)),
        ram=float(get("ram", 0.0)),
        ram_used_gb=float(get("ram_used_gb", 0.0)),
        ram_total_gb=float(ram_total_gb or 0.0),
        swap=float(get("swap", 0.0)),
        disk_pct=float(get("disk_pct", 0.0)),
        disk_read_mbps=float(get("disk_read_mbps", 0.0)),
        disk_write_mbps=float(get("disk_write_mbps", 0.0)),
        net_sent_mbps=float(get("net_sent_mbps", 0.0)),
        net_recv_mbps=float(get("net_recv_mbps", 0.0)),
        load1_per_core=float(get("load1_per_core", 0.0)),
        proc_count=int(get("proc_count", 0)),
        cpu_freq_mhz=get("cpu_freq_mhz"),
        battery_pct=get("battery_pct"),
        battery_plugged=None if plugged is None else bool(plugged),
        temp_c=get("temp_c"),
        fan_rpm=get("fan_rpm"),
        uptime_seconds=float(get("uptime_seconds", 0.0)),
    )


def static_system_info() -> dict:
    """Hardware / OS facts that never change during a run (cheap to cache)."""
    import platform

    try:
        mem_total = psutil.virtual_memory().total / (1024**3)
    except Exception:
        mem_total = 0.0
    info = {
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor() or platform.machine(),
        "python": platform.python_version(),
        "cpu_logical": psutil.cpu_count(logical=True) or 1,
        "cpu_physical": psutil.cpu_count(logical=False) or 1,
        "ram_total_gb": round(mem_total, 2),
    }
    try:
        info["boot_time"] = psutil.boot_time()
    except Exception:
        info["boot_time"] = None
    return info
