"""Training callbacks."""

from __future__ import annotations

import csv
from pathlib import Path

from scdm.common.utils.io import save_json


class MetricHistoryCallback:
    """Collect and persist validation-checkpoint level metrics."""

    def __init__(self) -> None:
        self.records: list[dict] = []

    def on_epoch_end(self, metrics: dict) -> None:
        self.records.append(dict(metrics))

    def save(self, output_dir: str | Path) -> None:
        output_dir = Path(output_dir)
        save_json(output_dir / "train_log.json", self.records)

        if not self.records:
            return

        fieldnames = list(self.records[0].keys())
        with (output_dir / "loss_curve.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.records)
