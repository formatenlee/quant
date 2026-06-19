from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


@dataclass
class Config:
    data_dir: Path = PROJECT_ROOT / "data"
    request_delay: float = 2.0
    request_jitter: float = 0.8
    max_retries: int = 3
    retry_backoff: float = 8.0
    batch_pause_every: int = 40
    batch_pause_seconds: float = 25.0
    ban_consecutive_failures: int = 5
    ban_cooldown_seconds: float = 900.0
    fallback_delay: float = 20.0
    today_update_hour: int = 21
    timezone: str = "Asia/Shanghai"
    include_etf: bool = True
    include_bond_indices: bool = False
    em_index_categories: list[str] = field(
        default_factory=lambda: [
            "沪深重要指数",
            "上证系列指数",
            "深证系列指数",
            "中证系列指数",
        ]
    )
    include_stocks: bool = True
    stock_request_delay: float = 6.0
    stock_batch_pause_every: int = 30
    stock_batch_pause_seconds: float = 45.0
    incremental_overlap_days: int = 5
    adjust_mode: str = "qfq"
    large_cap_min_mv: float = 50_000_000_000.0
    sw_l2_mapping_refresh_days: int = 7
    sample_validation_count: int = 20
    qlib_staging_subdir: str = "qlib_staging"
    qlib_week_staging_subdir: str = "qlib_staging_week"
    qlib_min_staging_subdir: str = "qlib_staging_1min"
    qlib_deriv_staging_subdir: str = "qlib_staging_deriv"
    qlib_data_subdir: str = "qlib_data"
    derivatives_subdir: str = "derivatives"
    derivatives_futures_subdir: str = "derivatives/futures"
    indices_min_subdir: str = "indices_min"
    etf_min_subdir: str = "etf_min"
    minute_period: str = "5"
    minute_request_delay: float | None = None
    minute_data_source: str = "auto"  # sina | em | auto
    minute_max_retries: int = 4

    @property
    def meta_dir(self) -> Path:
        return self.data_dir / "meta"

    @property
    def indices_dir(self) -> Path:
        return self.data_dir / "indices"

    @property
    def etf_dir(self) -> Path:
        return self.data_dir / "etf"

    @property
    def stocks_dir(self) -> Path:
        return self.data_dir / "stocks"

    @property
    def universe_path(self) -> Path:
        return self.meta_dir / "universe.parquet"

    @property
    def qlib_staging_dir(self) -> Path:
        return self.data_dir / self.qlib_staging_subdir

    @property
    def qlib_week_staging_dir(self) -> Path:
        return self.data_dir / self.qlib_week_staging_subdir

    @property
    def qlib_min_staging_dir(self) -> Path:
        return self.data_dir / self.qlib_min_staging_subdir

    @property
    def qlib_deriv_staging_dir(self) -> Path:
        return self.data_dir / self.qlib_deriv_staging_subdir

    @property
    def derivatives_dir(self) -> Path:
        return self.data_dir / self.derivatives_subdir

    @property
    def derivatives_futures_dir(self) -> Path:
        return self.data_dir / self.derivatives_futures_subdir

    @property
    def derivatives_events_path(self) -> Path:
        return self.meta_dir / "derivatives_events.parquet"

    @property
    def qlib_data_dir(self) -> Path:
        return self.data_dir / self.qlib_data_subdir

    @property
    def indices_min_dir(self) -> Path:
        return self.data_dir / self.indices_min_subdir

    @property
    def etf_min_dir(self) -> Path:
        return self.data_dir / self.etf_min_subdir

    @property
    def effective_minute_delay(self) -> float:
        return self.minute_request_delay if self.minute_request_delay is not None else self.request_delay

    def ensure_dirs(self) -> None:
        for path in (
            self.meta_dir,
            self.indices_dir,
            self.etf_dir,
            self.stocks_dir,
            self.indices_min_dir,
            self.etf_min_dir,
            self.qlib_staging_dir,
            self.qlib_week_staging_dir,
            self.qlib_min_staging_dir,
            self.qlib_deriv_staging_dir,
            self.derivatives_dir,
            self.derivatives_futures_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


def load_config(path: Path | None = None) -> Config:
    config_path = path or DEFAULT_CONFIG_PATH
    raw: dict[str, Any] = {}
    if config_path.exists():
        with config_path.open(encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

    cfg = Config()
    if "data_dir" in raw:
        cfg.data_dir = Path(raw["data_dir"])
        if not cfg.data_dir.is_absolute():
            cfg.data_dir = PROJECT_ROOT / cfg.data_dir

    for key in (
        "request_delay",
        "request_jitter",
        "max_retries",
        "retry_backoff",
        "batch_pause_every",
        "batch_pause_seconds",
        "ban_consecutive_failures",
        "ban_cooldown_seconds",
        "fallback_delay",
        "today_update_hour",
        "timezone",
        "include_etf",
        "include_bond_indices",
        "em_index_categories",
        "include_stocks",
        "stock_request_delay",
        "stock_batch_pause_every",
        "stock_batch_pause_seconds",
        "incremental_overlap_days",
        "adjust_mode",
        "large_cap_min_mv",
        "sw_l2_mapping_refresh_days",
        "sample_validation_count",
        "qlib_staging_subdir",
        "qlib_week_staging_subdir",
        "qlib_min_staging_subdir",
        "qlib_deriv_staging_subdir",
        "qlib_data_subdir",
        "derivatives_subdir",
        "derivatives_futures_subdir",
        "indices_min_subdir",
        "etf_min_subdir",
        "minute_period",
        "minute_request_delay",
        "minute_data_source",
        "minute_max_retries",
    ):
        if key in raw:
            setattr(cfg, key, raw[key])

    return cfg
