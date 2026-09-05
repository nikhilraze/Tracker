"""Streamlit UI.

The UI is a pure reader. It never samples, never writes telemetry and never
blocks: the engine thread owns all of that. Live sections are wrapped in
``st.fragment(run_every=...)`` so only those blocks re-execute on a timer,
instead of the old ``time.sleep(5); st.rerun()`` which re-ran the entire script
(and re-logged a row) every few seconds.
"""

from __future__ import annotations

import time
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

from .collector import Sample
from .config import load_config
from .engine import MonitorEngine, Snapshot, get_engine

REFRESH_SECONDS = 1.0

SEVERITY_STYLE = {
    "critical": ("error", "🚨"),
    "warning": ("warning", "⚠️"),
    "info": ("info", "ℹ️"),
}


# ---------------------------------------------------------------------------
# engine wiring
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Starting the monitoring engine...")
def _engine() -> MonitorEngine:
    """One engine per Streamlit server process, shared by all browser sessions."""
    return get_engine(load_config())


def _fmt_duration(seconds: float | None) -> str:
    if seconds is None:
        return "n/a"
    seconds = int(max(0, seconds))
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    if days:
        return f"{days}d {hours}h {minutes}m"
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _fmt_skill(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.1%}"


def history_frame(samples: list[Sample]) -> pd.DataFrame:
    if not samples:
        return pd.DataFrame()
    frame = pd.DataFrame([s.as_dict() for s in samples])
    frame["time"] = pd.to_datetime(frame["ts"], unit="s")
    frame["disk_io_mbps"] = frame["disk_read_mbps"] + frame["disk_write_mbps"]
    frame["net_io_mbps"] = frame["net_sent_mbps"] + frame["net_recv_mbps"]
    return frame


# ---------------------------------------------------------------------------
# shared widgets
# ---------------------------------------------------------------------------
def render_alerts(snapshot: Snapshot) -> None:
    report = snapshot.feedback
    if report is None:
        return
    if not report.alerts:
        st.success("No active alerts.")
        return
    for alert in report.alerts:
        kind, icon = SEVERITY_STYLE.get(alert.severity, ("info", "•"))
        getattr(st, kind)(f"{icon} **{alert.title}** - {alert.detail}")


def render_advice(snapshot: Snapshot) -> None:
    report = snapshot.feedback
    if report is None or not report.advice:
        return
    for item in report.advice:
        _, icon = SEVERITY_STYLE.get(item.severity, ("info", "•"))
        st.markdown(f"{icon} {item.text}")
        if item.because:
            st.caption(f"why: {item.because}")


def render_status_bar(engine: MonitorEngine) -> None:
    status = engine.status()
    age = status["snapshot_age"]
    if not status["running"]:
        st.error("Sampler thread is not running.")
    elif age is not None and age > engine.config.sample_interval * 5:
        st.warning(f"Last sample was {age:.0f}s ago - the sampler may be stalled.")
    bits = [
        f"**{status['tick']}** samples this run",
        f"every **{status['sample_interval']:.0f}s**",
        f"buffer **{status['buffer']}**",
        f"db **{status['db_mb']:.1f} MB**",
    ]
    if status["skipped_ticks"]:
        bits.append(f"skipped **{status['skipped_ticks']}**")
    if status["errors"]:
        bits.append(f"errors **{status['errors']}**")
    st.caption(" · ".join(bits))


# ---------------------------------------------------------------------------
# pages
# ---------------------------------------------------------------------------
def page_live(engine: MonitorEngine) -> None:
    st.title("Live")
    st.caption(
        "Updates every second from a background sampler. Sampling continues even "
        "when this tab is closed."
    )

    @st.fragment(run_every=REFRESH_SECONDS)
    def live_block() -> None:
        snapshot = engine.snapshot()
        if snapshot is None:
            st.info("Collecting the first sample...")
            return
        sample = snapshot.sample

        cols = st.columns(4)
        cols[0].metric("CPU", f"{sample.cpu:.1f}%", f"hottest core {sample.cpu_max_core:.0f}%")
        cols[1].metric(
            "RAM",
            f"{sample.ram:.1f}%",
            f"{sample.ram_used_gb:.1f} / {sample.ram_total_gb:.1f} GB",
        )
        cols[2].metric("Disk used", f"{sample.disk_pct:.1f}%")
        cols[3].metric("Health", snapshot.health, help="100 = no resource under pressure")

        cols = st.columns(4)
        cols[0].metric("Disk I/O", f"{sample.disk_io_mbps:.2f} MB/s")
        cols[1].metric("Network", f"{sample.net_io_mbps:.2f} MB/s")
        cols[2].metric("Swap", f"{sample.swap:.1f}%")
        if sample.temp_c is not None:
            cols[3].metric("Temp", f"{sample.temp_c:.0f} °C")
        elif sample.battery_pct is not None:
            plugged = "charging" if sample.battery_plugged else "on battery"
            cols[3].metric("Battery", f"{sample.battery_pct:.0f}%", plugged)
        else:
            cols[3].metric("Processes", sample.proc_count)

        st.divider()
        left, right = st.columns([2, 1])

        with left:
            st.subheader("Last 5 minutes")
            frame = history_frame(engine.history(limit=300))
            if not frame.empty:
                st.line_chart(
                    frame.set_index("time")[["cpu", "ram"]],
                    height=260,
                )
                st.area_chart(
                    frame.set_index("time")[["disk_io_mbps", "net_io_mbps"]],
                    height=160,
                )

        with right:
            st.subheader("Forecast")
            horizon = engine.config.forecast_horizon
            for target, result in snapshot.forecasts.items():
                if result.predicted is None:
                    st.metric(f"{target.upper()} in {horizon}s", "warming up")
                    st.caption(f"{result.trained_on} labelled pairs so far")
                    continue
                current = sample.cpu if target == "cpu" else sample.ram
                st.metric(
                    f"{target.upper()} in {horizon}s",
                    f"{result.predicted:.1f}%",
                    f"{result.predicted - current:+.1f} pts",
                )
                st.caption(f"source: {result.source}")

            st.subheader("Anomaly")
            if snapshot.anomaly is not None:
                anomaly = snapshot.anomaly
                st.progress(
                    min(1.0, anomaly.score),
                    text=f"{anomaly.score:.2f} - {anomaly.label}",
                )
                if anomaly.tripped:
                    for name, z in anomaly.tripped[:3]:
                        st.caption(f"{name}: {z:+.1f} sigma from its usual level")
                elif not anomaly.forest_ready:
                    st.caption("IsolationForest still collecting its first window.")

        st.divider()
        st.subheader("What to do now")
        render_alerts(snapshot)
        render_advice(snapshot)
        render_status_bar(engine)

    live_block()


def page_processes(engine: MonitorEngine) -> None:
    st.title("Processes")
    st.caption(
        f"Rescanned every {engine.config.process_scan_every} samples - this is the "
        "expensive part of collection, so it runs slower than the 1 Hz metrics."
    )

    @st.fragment(run_every=max(2.0, engine.config.process_scan_every * REFRESH_SECONDS))
    def process_block() -> None:
        snapshot = engine.snapshot()
        if snapshot is None or not snapshot.processes:
            st.info("Waiting for the first process scan...")
            return

        cpu_count = engine.collector.cpu_count
        frame = pd.DataFrame([p.as_dict() for p in snapshot.processes])
        frame = frame.rename(
            columns={
                "pid": "PID",
                "name": "Process",
                "cpu_percent": "CPU %",
                "cpu_percent_normalized": f"CPU % (of {cpu_count} cores)",
                "memory_mb": "Memory MB",
                "memory_percent": "Memory %",
            }
        )
        st.dataframe(
            frame,
            hide_index=True,
            width="stretch",
            column_config={
                "CPU %": st.column_config.NumberColumn(format="%.1f"),
                f"CPU % (of {cpu_count} cores)": st.column_config.NumberColumn(format="%.1f"),
                "Memory MB": st.column_config.NumberColumn(format="%.0f"),
                "Memory %": st.column_config.NumberColumn(format="%.1f"),
            },
        )
        st.caption(
            "A process can exceed 100% CPU because it is measured across all cores. "
            "The normalised column is the share of total machine capacity."
        )

        top = snapshot.processes[0]
        st.info(
            f"Heaviest right now: **{top.name}** (PID {top.pid}) - "
            f"{top.cpu_percent:.0f}% CPU, {top.memory_mb:.0f} MB RAM"
        )

    process_block()


def page_analytics(engine: MonitorEngine) -> None:
    st.title("Analytics")
    storage = engine.storage

    tab_recent, tab_long, tab_habits = st.tabs(
        ["Recent (per second)", "Long range (per minute)", "Usage habits"]
    )

    with tab_recent:
        minutes = st.slider("Window (minutes)", 1, 60, 10, key="recent_window")
        frame = history_frame(engine.history(limit=int(minutes * 60 / engine.config.sample_interval)))
        if frame.empty:
            st.info("No samples yet.")
        else:
            cols = st.columns(4)
            cols[0].metric("Samples", len(frame))
            cols[1].metric("Avg CPU", f"{frame['cpu'].mean():.1f}%")
            cols[2].metric("Peak CPU", f"{frame['cpu'].max():.1f}%")
            cols[3].metric("Avg RAM", f"{frame['ram'].mean():.1f}%")

            metrics = st.multiselect(
                "Metrics",
                ["cpu", "ram", "swap", "disk_io_mbps", "net_io_mbps", "health_score"]
                if "health_score" in frame
                else ["cpu", "ram", "swap", "disk_io_mbps", "net_io_mbps"],
                default=["cpu", "ram"],
                key="recent_metrics",
            )
            if metrics:
                st.line_chart(frame.set_index("time")[metrics], height=320)

            st.subheader("Distribution")
            metric = st.selectbox("Metric", ["cpu", "ram", "swap"], key="dist_metric")
            counts, edges = np.histogram(frame[metric].dropna(), bins=20, range=(0, 100))
            hist = pd.DataFrame(
                {"count": counts},
                index=[f"{edges[i]:.0f}-{edges[i+1]:.0f}%" for i in range(len(counts))],
            )
            st.bar_chart(hist, height=240)

    with tab_long:
        hours = st.slider("Window (hours)", 1, 168, 24, key="long_window")
        rows = storage.rollups_since(time.time() - hours * 3600)
        if not rows:
            st.info(
                "No per-minute rollups yet. They are built once a minute has fully "
                "elapsed, so leave the app running for a few minutes."
            )
        else:
            frame = pd.DataFrame([dict(r) for r in rows])
            frame["time"] = pd.to_datetime(frame["bucket_ts"], unit="s")
            st.caption(
                f"{len(frame)} one-minute buckets. Raw per-second rows are kept for "
                f"{engine.config.raw_retention_hours:.0f}h; these rollups for "
                f"{engine.config.rollup_retention_days:.0f} days."
            )
            st.line_chart(frame.set_index("time")[["cpu_avg", "cpu_max", "ram_avg"]], height=300)
            available = [c for c in ("disk_io_mbps_avg", "net_io_mbps_avg") if c in frame]
            if available:
                st.area_chart(frame.set_index("time")[available], height=200)
            if frame["anomaly_max"].notna().any():
                st.subheader("Peak anomaly score per minute")
                st.bar_chart(frame.set_index("time")[["anomaly_max"]], height=180)

    with tab_habits:
        st.subheader("Active time per day")
        st.caption(
            "Activity is inferred from resource usage, not from keyboard or mouse "
            "input, so a long build with nobody at the desk still counts as active."
        )
        daily = storage.daily_activity(days=14)
        if not daily:
            st.info("Not enough history yet.")
        else:
            frame = pd.DataFrame([dict(r) for r in daily])
            frame["active_hours"] = frame["active_seconds"].fillna(0) / 3600
            frame["observed_hours"] = frame["observed_seconds"].fillna(0) / 3600
            st.bar_chart(frame.set_index("day")[["active_hours", "observed_hours"]], height=260)

        st.subheader("Load by hour of day")
        hourly = storage.hourly_profile(days=14)
        if hourly:
            frame = pd.DataFrame([dict(r) for r in hourly])
            st.bar_chart(frame.set_index("hour")[["cpu_avg", "ram_avg"]], height=240)
            busiest = frame.loc[frame["cpu_avg"].idxmax()]
            st.info(
                f"Your busiest hour is **{int(busiest['hour']):02d}:00** at "
                f"{busiest['cpu_avg']:.0f}% average CPU."
            )

        st.subheader("Recent activity sessions")
        sessions = storage.recent_sessions(limit=15)
        if sessions:
            rows = []
            for row in sessions:
                rows.append(
                    {
                        "Started": datetime.fromtimestamp(row["start_ts"]).strftime("%Y-%m-%d %H:%M"),
                        "Active min": round((row["active_seconds"] or 0) / 60, 1),
                        "Avg CPU %": round(row["cpu_avg"] or 0, 1),
                        "Avg RAM %": round(row["ram_avg"] or 0, 1),
                        "Open": "no" if row["closed"] else "yes",
                    }
                )
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        else:
            st.info("No sessions recorded yet.")


def page_insights(engine: MonitorEngine) -> None:
    st.title("Insights")

    @st.fragment(run_every=2.0)
    def insight_block() -> None:
        snapshot = engine.snapshot()
        if snapshot is None:
            st.info("Waiting for data...")
            return
        report = snapshot.feedback

        cols = st.columns(3)
        cols[0].metric("Health", snapshot.health)
        cols[1].metric(
            "Anomaly", f"{snapshot.anomaly.score:.2f}" if snapshot.anomaly else "n/a",
            snapshot.anomaly.label if snapshot.anomaly else None,
        )
        cols[2].metric("Uptime", _fmt_duration(snapshot.sample.uptime_seconds))

        st.subheader("Active alerts")
        render_alerts(snapshot)

        st.subheader("Recommended actions")
        render_advice(snapshot)

        if report is not None:
            projections = []
            if report.ram_exhaustion_minutes is not None:
                projections.append(
                    f"At the current trend RAM would reach 95% in about "
                    f"**{report.ram_exhaustion_minutes:.0f} minutes**."
                )
            if report.disk_exhaustion_hours is not None:
                projections.append(
                    f"Disk would fill in about **{report.disk_exhaustion_hours:.0f} hours** "
                    "at the current write rate."
                )
            if projections:
                st.subheader("Projections")
                for text in projections:
                    st.markdown(f"- {text}")
                st.caption(
                    "Straight-line extrapolation of the recent trend - useful for "
                    "catching leaks, not a guarantee."
                )

        if snapshot.session is not None:
            session = snapshot.session
            st.subheader("This session")
            cols = st.columns(3)
            cols[0].metric("State", "active" if session.active else "idle")
            cols[1].metric("Continuous", f"{session.continuous_minutes:.0f} min")
            cols[2].metric("Active today", _fmt_duration(session.active_seconds_today))
            if session.break_due:
                st.warning("You have been going for a while - worth a short break.")

    insight_block()

    st.divider()
    st.subheader("Event history")
    events = engine.storage.recent_events(limit=100)
    if not events:
        st.info("No events recorded yet. Alerts, anomalies and sessions appear here.")
        return
    rows = [
        {
            "When": datetime.fromtimestamp(row["ts"]).strftime("%Y-%m-%d %H:%M:%S"),
            "Severity": row["severity"],
            "Kind": row["kind"],
            "Message": row["message"],
        }
        for row in events
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", height=320)


def page_model(engine: MonitorEngine) -> None:
    st.title("Model")
    st.caption(
        "Both models are trained from scratch on this machine, from the samples "
        "this app collects. Nothing is downloaded and nothing is uploaded."
    )

    snapshot = engine.snapshot()
    horizon = engine.config.forecast_horizon

    st.subheader("1. Forecaster - supervised, self-labelled")
    st.markdown(
        f"""
Predicts CPU and RAM **{horizon} seconds ahead**. Labels cost nothing: a
prediction made at time *t* is compared against reality at *t+{horizon}*, and
that pair becomes a training example. The model learns the **change**
(`y[t+{horizon}] - y[t]`), so zero weights are exactly equal to the naive
"nothing will change" baseline.
"""
    )

    if snapshot is None:
        st.info("Waiting for the first sample...")
    else:
        for target, result in snapshot.forecasts.items():
            with st.container(border=True):
                st.markdown(f"**{target.upper()}**")
                cols = st.columns(4)
                cols[0].metric("Labelled pairs", result.trained_on)
                cols[1].metric("Refits", result.fit_count)
                cols[2].metric(
                    "Model error (MAE)",
                    "n/a" if result.mae is None else f"{result.mae:.2f}",
                )
                cols[3].metric(
                    "Baseline error (MAE)",
                    "n/a" if result.naive_mae is None else f"{result.naive_mae:.2f}",
                )

                cols = st.columns(3)
                cols[0].metric("Skill vs baseline", _fmt_skill(result.skill))
                cols[1].metric("Trust weight", f"{result.trust_weight:.0%}")
                cols[2].metric("Emitted skill", _fmt_skill(result.effective_skill))

                if result.skill is None:
                    st.caption("Not enough graded predictions yet.")
                elif result.skill > 0:
                    st.success(
                        f"Beating the baseline by {result.skill:.1%}, so the forecast "
                        f"leans on the model ({result.trust_weight:.0%})."
                    )
                else:
                    st.warning(
                        "Not beating the naive baseline, so the forecast is shrunk back "
                        "toward 'no change'. This is expected for second-scale CPU, "
                        "which is close to unpredictable; RAM usually does better."
                    )

        st.subheader("What the model is keying on")
        target = st.selectbox("Target", list(snapshot.forecasts.keys()), key="weight_target")
        weights = engine.forecast.forecasters[target].top_weights(12)
        if not weights:
            st.info("No fitted coefficients yet.")
        else:
            frame = pd.DataFrame(weights, columns=["feature", "weight"]).set_index("feature")
            st.bar_chart(frame, height=320)
            st.caption(
                "Ridge coefficients on standardised features, predicting the change in "
                "the target. Sign and size show direction and influence, not causation."
            )

    st.divider()
    st.subheader("Prediction audit trail")
    audit_target = st.selectbox("Metric", ["cpu", "ram"], key="audit_target")
    rows = engine.storage.prediction_scores(audit_target, limit=400)
    if not rows:
        st.info("No graded predictions stored yet.")
    else:
        frame = pd.DataFrame([dict(r) for r in rows])
        frame["time"] = pd.to_datetime(frame["ts"], unit="s")
        st.line_chart(frame.set_index("time")[["predicted", "actual"]], height=280)
        cols = st.columns(3)
        cols[0].metric("Graded", len(frame))
        cols[1].metric("Model MAE", f"{frame['abs_err'].mean():.2f}")
        cols[2].metric("Baseline MAE", f"{frame['naive_abs_err'].mean():.2f}")
        st.caption(
            "Every prediction is stored with the value that actually occurred, so the "
            "accuracy figures above are auditable rather than asserted."
        )

    st.divider()
    st.subheader("2. Anomaly detector - unsupervised, two layers")
    st.markdown(
        """
* **Robust z-scores** per metric, updated every sample from an exponentially
  weighted mean and mean-absolute-deviation. Usable within ~30 seconds and
  immune to a single huge spike inflating the threshold.
* **IsolationForest** refit on a rolling window, which catches abnormal
  *combinations* of metrics that per-metric thresholds miss - for example
  moderate CPU with unusual disk and network at the same time.

The final score is the higher of the two, mapped to 0-1.
"""
    )
    if snapshot is not None and snapshot.anomaly is not None:
        anomaly = snapshot.anomaly
        cols = st.columns(4)
        cols[0].metric("Score", f"{anomaly.score:.2f}", anomaly.label)
        cols[1].metric("Window", anomaly.window_size)
        cols[2].metric("Refits", anomaly.train_count)
        cols[3].metric(
            "Last refit",
            "never" if anomaly.last_trained is None
            else f"{time.time() - anomaly.last_trained:.0f}s ago",
        )
        cols = st.columns(2)
        cols[0].metric(
            "z-score layer",
            "n/a" if anomaly.zscore_score is None else f"{anomaly.zscore_score:.2f}",
        )
        cols[1].metric(
            "forest layer",
            "n/a" if anomaly.forest_score is None else f"{anomaly.forest_score:.2f}",
        )

    status = engine.status()
    st.divider()
    st.subheader("Checkpointing")
    st.write(
        f"Models are checkpointed to `{engine.config.model_dir}` every "
        f"{engine.config.model_save_interval:.0f}s (and on shutdown), then reloaded at "
        "startup, so learning survives restarts."
    )
    st.json(
        {
            "restored_on_startup": status["models_restored"],
            "stale_labels_dropped": status["stale_labels_dropped"],
            "forecast_horizon_seconds": horizon,
            "rolling_window_samples": engine.config.forecast_window,
            "refit_every_samples": engine.config.forecast_refit_every,
        }
    )


def page_settings(engine: MonitorEngine) -> None:
    st.title("Settings & diagnostics")

    status = engine.status()
    cols = st.columns(4)
    cols[0].metric("Sampler", "running" if status["running"] else "stopped")
    cols[1].metric("Samples this run", status["tick"])
    cols[2].metric("Engine uptime", _fmt_duration(status["uptime_seconds"]))
    cols[3].metric("Database", f"{status['db_mb']:.1f} MB")

    if status["last_error"]:
        st.error(f"Last error: {status['last_error']} (total {status['errors']})")
    if status["skipped_ticks"]:
        st.warning(
            f"{status['skipped_ticks']} ticks were skipped. This normally means the "
            "machine slept or was heavily loaded; the sampler skips ahead instead of "
            "firing a burst of catch-up samples."
        )

    st.subheader("This machine")
    st.json(engine.system_info)

    st.subheader("Storage")
    cols = st.columns(3)
    cols[0].metric("Raw samples", engine.storage.sample_count())
    cols[1].metric("Observed minutes", f"{engine.storage.total_sampled_seconds()/60:.0f}")
    cols[2].metric("Data dir", "")
    st.code(str(engine.config.data_dir))

    left, right = st.columns(2)
    if left.button("Build rollups now", width="stretch"):
        made = engine.storage.build_rollups()
        st.success(f"Rebuilt {made} minute buckets.")
    if right.button("Run retention cleanup", width="stretch"):
        deleted = engine.storage.prune(
            engine.config.raw_retention_hours,
            engine.config.rollup_retention_days,
            engine.config.event_retention_days,
            engine.config.prediction_retention_hours,
        )
        st.success(f"Deleted: {deleted}")

    st.subheader("Export")
    st.caption("Downloads the in-memory buffer as CSV for external analysis.")
    frame = history_frame(engine.history())
    if not frame.empty:
        st.download_button(
            "Download recent samples (CSV)",
            frame.to_csv(index=False).encode(),
            file_name=f"neurotrack_{int(time.time())}.csv",
            mime="text/csv",
        )

    st.subheader("Configuration")
    st.caption(
        "Values come from defaults plus NEUROTRACK_* environment variables. Restart "
        "the app after changing them, since the sampler and models are built at start."
    )
    config = engine.config.as_dict()
    st.dataframe(
        pd.DataFrame(
            {"setting": list(config.keys()), "value": [str(v) for v in config.values()]}
        ),
        hide_index=True,
        width="stretch",
        height=420,
    )


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
PAGES = {
    "Live": page_live,
    "Processes": page_processes,
    "Analytics": page_analytics,
    "Insights": page_insights,
    "Model": page_model,
    "Settings": page_settings,
}


def main() -> None:
    st.set_page_config(page_title="NeuroTrack AI", page_icon="🧠", layout="wide")
    engine = _engine()

    with st.sidebar:
        st.title("🧠 NeuroTrack AI")
        choice = st.radio("View", list(PAGES), label_visibility="collapsed")
        st.divider()
        snapshot = engine.snapshot()
        if snapshot is not None:
            st.metric("Health", snapshot.health)
            st.metric("CPU", f"{snapshot.sample.cpu:.0f}%")
            st.metric("RAM", f"{snapshot.sample.ram:.0f}%")
            if snapshot.feedback and snapshot.feedback.alerts:
                st.error(f"{len(snapshot.feedback.alerts)} active alert(s)")
        st.divider()
        st.caption(
            "All data stays on this machine in "
            f"`{engine.config.data_dir}`."
        )

    PAGES[choice](engine)


if __name__ == "__main__":
    main()
