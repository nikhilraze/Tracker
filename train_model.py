"""Offline training and evaluation for the NeuroTrack forecaster.

The live app trains itself continuously, so this tool is not required. It exists
for three things the live loop cannot do:

1. **Audit.** Replay your stored history through the production training loop and
   print how the model performed against the naive baseline. Because the replay
   uses the same ``ForecastService`` the app uses, the numbers are directly
   comparable rather than a separate research script.
2. **Warm start.** Train on existing history and checkpoint the result, so a
   fresh install predicts usefully immediately instead of spending minutes
   shrunk to the baseline.
3. **Horizon choice.** Report skill at several horizons so you can pick one that
   is actually predictable on *your* machine.

Usage:
    python train_model.py                        # evaluate at the configured horizon
    python train_model.py --save                 # evaluate, then checkpoint
    python train_model.py --horizons 3,5,15,30   # compare horizons
    python train_model.py --limit 50000          # cap samples used
"""

from __future__ import annotations

import argparse

import numpy as np

from neurotrack.collector import sample_from_row, static_system_info
from neurotrack.config import load_config
from neurotrack.features import MIN_HISTORY, build_forecast_features
from neurotrack.models import ForecastService
from neurotrack.storage import Storage

# Graded predictions ignored at the start, while the model has not fitted yet.
WARMUP_PREDICTIONS = 200


def load_history(storage: Storage, limit: int | None) -> list:
    ram_total = static_system_info().get("ram_total_gb", 0.0)
    rows = storage.recent_samples(limit=limit or 10_000_000)
    samples = []
    for row in rows:
        try:
            samples.append(sample_from_row(row, ram_total))
        except Exception:
            continue
    samples.sort(key=lambda s: s.ts)
    return samples


def replay(samples: list, horizon: int, config, buffer_size: int = 3600):
    """Run stored history through the real training loop.

    Returns (service, per-target stats). Uses the production ``ForecastService``
    so the label delay, shrinkage and metrics are exactly the live behaviour.
    """
    service = ForecastService(
        horizon=horizon,
        decay=config.metric_decay,
        warmup=config.model_warmup_samples,
        sample_interval=config.sample_interval,
        window=config.forecast_window,
        refit_every=config.forecast_refit_every,
        alpha=config.forecast_alpha,
        min_fit=config.forecast_min_fit,
    )

    history: list = []
    for sample in samples:
        history.append(sample)
        if len(history) > buffer_size:
            history.pop(0)
        features = build_forecast_features(history, len(history) - 1)
        service.observe(features, {"cpu": sample.cpu, "ram": sample.ram}, sample.ts)

    graded: dict[str, list[tuple[float, float]]] = {}
    for _ts, target, _h, _pred, _truth, model_err, naive_err in service.graded_rows:
        graded.setdefault(target, []).append((model_err, naive_err))

    stats = {}
    for target, rows in graded.items():
        scored = rows[WARMUP_PREDICTIONS:]
        if not scored:
            stats[target] = None
            continue
        model_mae = float(np.mean([r[0] for r in scored]))
        naive_mae = float(np.mean([r[1] for r in scored]))
        stats[target] = {
            "n": len(scored),
            "mae": model_mae,
            "naive_mae": naive_mae,
            "skill": 1 - model_mae / naive_mae if naive_mae > 0 else None,
            "raw_skill": service.forecasters[target].error.skill,
            "trust": service.forecasters[target].trust_weight,
            "refits": service.forecasters[target].fit_count,
        }
    return service, stats


def print_stats(horizon: int, stats: dict) -> None:
    print(f"\nhorizon = {horizon}s")
    print(f"  {'target':<6} {'graded':>7} {'MAE':>8} {'baseline':>9} {'skill':>8} {'trust':>6} {'refits':>7}")
    print("  " + "-" * 58)
    for target, row in sorted(stats.items()):
        if row is None:
            print(f"  {target:<6} {'-':>7} {'not enough data':>36}")
            continue
        skill = "n/a" if row["skill"] is None else f"{row['skill']:+.1%}"
        print(
            f"  {target:<6} {row['n']:>7} {row['mae']:>8.3f} {row['naive_mae']:>9.3f}"
            f" {skill:>8} {row['trust']:>6.0%} {row['refits']:>7}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the NeuroTrack forecaster on stored history."
    )
    parser.add_argument("--save", action="store_true", help="checkpoint the trained model")
    parser.add_argument("--limit", type=int, default=None, help="max samples to replay")
    parser.add_argument(
        "--horizons",
        type=str,
        default=None,
        help="comma-separated horizons to compare, e.g. 3,5,15,30",
    )
    args = parser.parse_args(argv)

    config = load_config()
    storage = Storage(config.db_path)

    print(f"database: {config.db_path}")
    samples = load_history(storage, args.limit)
    print(f"loaded {len(samples)} stored samples")

    minimum = MIN_HISTORY + config.forecast_min_fit + config.forecast_horizon + WARMUP_PREDICTIONS
    if len(samples) < minimum:
        print(
            f"\nNot enough history to evaluate: need at least ~{minimum} samples, "
            f"have {len(samples)}."
        )
        print(
            "Leave the dashboard running for a while (raw samples are retained for "
            f"{config.raw_retention_hours:.0f}h) and try again."
        )
        storage.close()
        return 1

    span_hours = (samples[-1].ts - samples[0].ts) / 3600
    print(f"span: {span_hours:.2f} hours")

    if args.horizons:
        horizons = [int(h.strip()) for h in args.horizons.split(",") if h.strip()]
        results = {}
        for horizon in horizons:
            _service, stats = replay(samples, horizon, config)
            results[horizon] = stats
            print_stats(horizon, stats)

        print("\nsummary - positive skill means the model beats 'nothing will change':")
        for target in ("cpu", "ram"):
            best = None
            for horizon, stats in results.items():
                row = stats.get(target)
                if row and row["skill"] is not None:
                    if best is None or row["skill"] > best[1]:
                        best = (horizon, row["skill"])
            if best:
                # A couple of percent is inside run-to-run noise; do not sell
                # that as a working forecast.
                if best[1] >= 0.05:
                    verdict = f"genuinely forecastable - consider horizon {best[0]}s"
                elif best[1] > 0:
                    verdict = "only marginal, essentially matches the baseline"
                else:
                    verdict = "not forecastable at any tested horizon"
                print(f"  {target}: best {best[1]:+.1%} at {best[0]}s -> {verdict}")
        storage.close()
        return 0

    service, stats = replay(samples, config.forecast_horizon, config)
    print_stats(config.forecast_horizon, stats)

    for target, row in stats.items():
        if row is None or row["skill"] is None:
            continue
        if row["skill"] > 0:
            print(
                f"\n{target}: model is genuinely better than the baseline "
                f"({row['skill']:+.1%}); the live app will lean on it."
            )
        else:
            print(
                f"\n{target}: model does not beat the baseline here, so the live app "
                "shrinks its forecast back toward 'no change'. That is the intended "
                "safety behaviour, not a bug."
            )

    if args.save:
        config.model_dir.mkdir(parents=True, exist_ok=True)
        service.save(config.forecaster_path)
        print(f"\nsaved forecaster to {config.forecaster_path}")
        print("The dashboard will pick this up on its next start.")
    else:
        print("\n(run with --save to checkpoint this model for the live app)")

    storage.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
