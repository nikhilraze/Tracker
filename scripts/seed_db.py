"""Seed a database with a synthetic trace, to exercise train_model.py end to end.

Development aid only - the real app never writes synthetic data.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from neurotrack.models import health_score
from neurotrack.storage import Storage
from smoke_models import make_trace


def main() -> int:
    target = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/nt_seed")
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 4000

    storage = Storage(target / "neurotrack.db")
    rows = []
    for sample in make_trace(count):
        row = sample.as_dict()
        row["anomaly_score"] = 0.0
        row["health_score"] = health_score(
            sample.cpu, sample.ram, sample.disk_pct, sample.swap, sample.temp_c
        )
        row["active"] = 1 if sample.cpu > 12 else 0
        rows.append(row)

    inserted = storage.insert_samples(rows)
    storage.build_rollups()
    print(f"seeded {inserted} samples into {target/'neurotrack.db'}")
    print(f"rollups: {len(storage.rollups_since(0))}")
    storage.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
