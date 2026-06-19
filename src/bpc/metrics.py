"""Structured training metrics logger (JSONL + CSV)."""

from __future__ import annotations

import csv
import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


class MetricsLogger:
    """Append-only metrics sink for monitoring loss, grad_norm, lr, etc."""

    def __init__(self, log_dir: Path):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.log_dir / "metrics.jsonl"
        self.csv_path = self.log_dir / "metrics.csv"
        self._csv_fields: list[str] | None = None
        self._csv_file = None
        self._csv_writer = None

    def log(self, record: dict[str, Any]) -> None:
        row = {"timestamp": datetime.now(timezone.utc).isoformat(), **record}
        safe_row = {k: _json_safe(v) for k, v in row.items()}
        with self.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(safe_row, ensure_ascii=False) + "\n")

        if self._csv_writer is None:
            self._csv_fields = list(row.keys())
            self._csv_file = self.csv_path.open("w", encoding="utf-8", newline="")
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=self._csv_fields)
            self._csv_writer.writeheader()
        else:
            new_keys = [k for k in row if k not in self._csv_fields]
            if new_keys:
                self._csv_file.close()
                self._csv_fields.extend(new_keys)
                existing_rows: list[dict] = []
                if self.csv_path.exists():
                    with self.csv_path.open(encoding="utf-8") as rf:
                        existing_rows = list(csv.DictReader(rf))
                self._csv_file = self.csv_path.open("w", encoding="utf-8", newline="")
                self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=self._csv_fields)
                self._csv_writer.writeheader()
                for r in existing_rows:
                    self._csv_writer.writerow({k: r.get(k, "") for k in self._csv_fields})

        self._csv_writer.writerow({k: row.get(k, "") for k in self._csv_fields})
        self._csv_file.flush()

    def close(self) -> None:
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None
