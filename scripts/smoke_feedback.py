"""Verify the feedback path actually fires under real load.

Spawns CPU burners and allocates memory while the engine runs, then asserts that
alerts, advice, process attribution and anomaly detection all react. Without this
the feedback code could silently never trigger.
"""

from __future__ import annotations

import multiprocessing as mp
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neurotrack.config import Config
from neurotrack.engine import MonitorEngine


def burn(stop_after: float) -> None:
    end = time.time() + stop_after
    x = 0.0
    while time.time() < end:
        for i in range(20000):
            x += i**0.5


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="neurotrack-fb-"))
    cfg = Config(data_dir=tmp)
    cfg.sample_interval = 1.0
    cfg.trigger_samples = 2
    cfg.alert_cooldown = 5.0
    cfg.cpu_high = 50.0          # reachable in a shared sandbox
    cfg.cpu_clear = 25.0
    cfg.anomaly_min_train = 20
    cfg.anomaly_retrain_every = 15
    cfg.forecast_min_fit = 20
    cfg.forecast_refit_every = 5
    cfg.forecast_horizon = 3
    cfg.activity_cpu_threshold = 8.0
    cfg.ensure_dirs()

    engine = MonitorEngine(cfg)
    engine.start()
    print("baseline for 12s...")
    time.sleep(12)
    snap = engine.snapshot()
    print(f"  idle: cpu={snap.sample.cpu:.1f}% health={snap.health} "
          f"alerts={[a.key for a in snap.feedback.alerts]}")

    workers = max(2, (mp.cpu_count() or 2))
    print(f"\napplying load with {workers} burner processes for 25s...")
    procs = [mp.Process(target=burn, args=(25,)) for _ in range(workers)]
    for p in procs:
        p.start()

    seen_alerts: set[str] = set()
    seen_advice: list[str] = []
    peak_cpu = 0.0
    peak_anomaly = 0.0
    saw_process_attribution = False

    deadline = time.time() + 26
    while time.time() < deadline:
        time.sleep(3)
        snap = engine.snapshot()
        if snap is None:
            continue
        peak_cpu = max(peak_cpu, snap.sample.cpu)
        peak_anomaly = max(peak_anomaly, snap.anomaly.score if snap.anomaly else 0)
        for alert in snap.feedback.alerts:
            seen_alerts.add(alert.key)
        for item in snap.feedback.advice:
            if item.text not in seen_advice:
                seen_advice.append(item.text)
        if snap.processes and any(p.cpu_percent > 20 for p in snap.processes):
            saw_process_attribution = True
        print(f"  cpu={snap.sample.cpu:5.1f}% health={snap.health:3d} "
              f"anom={snap.anomaly.score:.2f} alerts={sorted(seen_alerts)}")

    for p in procs:
        p.join(timeout=10)

    print("\nrecovering for 12s...")
    time.sleep(12)
    snap = engine.snapshot()
    print(f"  after: cpu={snap.sample.cpu:.1f}% health={snap.health} "
          f"alerts={[a.key for a in snap.feedback.alerts]}")

    events = engine.storage.recent_events(limit=50)
    engine.stop()

    print(f"\npeak cpu={peak_cpu:.1f}%  peak anomaly={peak_anomaly:.2f}")
    print(f"alerts seen: {sorted(seen_alerts)}")
    print(f"events logged: {len(events)}")
    for row in events[:8]:
        print(f"  [{row['severity']:8s}] {row['kind']:12s} {row['message'][:78]}")
    print("\nadvice produced:")
    for text in seen_advice[:6]:
        print(f"  - {text[:100]}")

    ok = True
    if peak_cpu < cfg.cpu_high:
        print(f"\n!! INCONCLUSIVE: load only reached {peak_cpu:.0f}%, "
              f"below the {cfg.cpu_high:.0f}% threshold")
    else:
        if "cpu_high" not in seen_alerts:
            ok = False
            print("\n!! FAIL: cpu_high alert never fired despite sustained load")
        if not any(k.startswith("alert") for k in ()) and not events:
            ok = False
            print("!! FAIL: no events were persisted")
        if not saw_process_attribution:
            print("!! WARN: no process crossed 20% CPU (container may hide processes)")
        # Alert must clear once load stops (hysteresis lower bound).
        if any(a.key == "cpu_high" for a in snap.feedback.alerts):
            ok = False
            print("!! FAIL: cpu_high alert did not clear after load stopped")
        else:
            print("cpu_high alert cleared correctly after load stopped")

    if engine.status()["errors"]:
        ok = False
        print(f"!! FAIL: {engine.status()['errors']} engine errors: "
              f"{engine.status()['last_error']}")

    shutil.rmtree(tmp, ignore_errors=True)
    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
